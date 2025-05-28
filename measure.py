import os
import torch
import glob
import cv2
import lpips
import numpy as np
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def ssim(prediction, target):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr


import os
import glob
import cv2
import lpips
import numpy as np
from PIL import Image


def metrics(im_dir, label_dir):
    # 1) glob 결과 확인 및 예외 처리
    files = sorted(glob.glob(im_dir))
    if not files:
        raise RuntimeError(f"이미지 파일을 찾지 못했습니다: {im_dir}")

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    loss_fn = lpips.LPIPS(net='alex').cuda()
    n = 0

    for item in files:
        n += 1
        # --- PIL로 원본/타깃 이미지 로드 ---
        im1 = Image.open(item).convert('RGB')
        name = os.path.basename(item)  # "1.png" 등 파일명만 추출
        label_path = os.path.join(label_dir, name)  # "img/reference/1.png"
        im2 = Image.open(label_path).convert('RGB')

        # --- PSNR, SSIM 계산 (calculate_psnr, calculate_ssim 호출) ---
        im1_np = np.array(im1, dtype=np.float32)
        im2_np = np.array(im2, dtype=np.float32)
        score_psnr = calculate_psnr(im1_np, im2_np)
        score_ssim = calculate_ssim(im1_np, im2_np)

        # --- LPIPS 계산 ---
        # lpips.load_image은 HWC, [0,1] 범위 numpy 반환
        w, h = im2.size
        p0_img = lpips.load_image(item)
        ref_img = lpips.load_image(label_path)
        p0_resized = cv2.resize(p0_img, (h, w))
        ref_resized = cv2.resize(ref_img, (h, w))

        ex_p0 = lpips.im2tensor(p0_resized).cuda()
        ex_ref = lpips.im2tensor(ref_resized).cuda()
        score_lpips = loss_fn(ex_ref, ex_p0).item()

        # --- 합산 ---
        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_lpips += score_lpips

    # 2) 평균 계산
    avg_psnr /= n
    avg_ssim /= n
    avg_lpips /= n

    return avg_psnr, avg_ssim, avg_lpips


if __name__ == '__main__':
    im_dir = 'img/results/I/*.png'
    label_dir = 'img/reference'

    avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir)
    print(f"===> Avg.PSNR: {avg_psnr:.4f} dB")
    print(f"===> Avg.SSIM: {avg_ssim:.4f}")
    print(f"===> Avg.LPIPS: {avg_lpips:.4f}")
