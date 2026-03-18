import cv2
import numpy as np
import os

def apply_cartoon_rendering_comparison(image_path, save_name="cartoon_result.jpg"):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: '{image_path}' 파일을 찾을 수 없습니다.")
        return

    # 2. Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply median blur to reduce noise (자료 설정: ksize=5)
    gray = cv2.medianBlur(gray, 5)
    
    # 4. Detect edges using adaptive thresholding
    # 자료 설정: maxValue=255, adaptiveMethod=MEAN_C, thresholdType=BINARY, blockSize=9, C=9
    # C값이 9로 설정되어 매우 세밀하고 깔끔한 에지가 추출됩니다.
    edges = cv2.adaptiveThreshold(gray, 255, 
                                  cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 9, 9)

    # 5. Convert the image to color (Bilateral Filter로 색 단순화)
    # 자료 설정: d=9, sigmaColor=300, sigmaSpace=300
    # 색상 단순화 효과가 매우 강력하여 평면적인 느낌을 줍니다.
    color = cv2.bilateralFilter(img, 9, 300, 300)

    # 6. Combine the color image with the edges mask
    # 색상 이미지 위에 에지 마스크(검은 선)를 입힙니다.
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    # --- 추가된 단계: 원본과 결과 나란히 붙이기 (Comparison View) ---
    # 두 이미지의 크기가 같아야 붙일 수 있습니다. (img와 cartoon은 크기가 같습니다)
    # cv2.hconcat은 가로로(Horizontal) 두 이미지를 연결합니다.
    comparison = cv2.hconcat([img, cartoon])

    # 7. Display and Save
    # 비교 이미지(comparison)를 하나의 창에 출력합니다.
    cv2.imshow("Comparison (Original vs Cartoon Rendering)", comparison)
    
    # 각각 저장 및 비교샷 저장 (README.md 제출용)
    cv2.imwrite(save_name, cartoon)
    cv2.imwrite("comparison_view.jpg", comparison) # 비교 샷도 저장
    
    print(f"성공적으로 '{save_name}'으로 저장되었습니다.")
    print(f"비교 이미지가 'comparison_view.jpg'로 저장되었습니다.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 실행부
if __name__ == "__main__":
    # 이미지 파일 경로 (폴더 구조에 맞춰 수정하세요)
    input_file = 'images/input1.jpg' 
    
    if os.path.exists(input_file):
        apply_cartoon_rendering_comparison(input_file)
    else:
        print(f"파일을 찾을 수 없습니다: {input_file}")
        print("이미지 파일이 'images' 폴더 안에 있는지 확인해 주세요.")