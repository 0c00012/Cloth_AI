## 전체 파이프라인

1. **SAM(세그멘테이션에 특화된 딥러닝 모델)을 사용하여 사진에서 옷 부분만 crop**

![output.png](attachment:edcfe438-62bb-4689-80ee-d9062f15c02d:output.png)

1. **4가지 이미지를 Blender를 통해 다양한 조합으로 스와치를 제작**
    1. 6가지 직조 방법을 사용하여 출력 
    2. 경사, 위사등의 실 합성순서, 크기 등을 랜덤하게 바꿔가며 출력
    
    ![20231024111228641d769538bc4297ae63a2228f8246d9.webp](attachment:f8cac911-3ae5-40ee-8381-08ce90ed9f86:20231024111228641d769538bc4297ae63a2228f8246d9.webp)
    
    ![image.png](attachment:82a523e8-87f1-40c1-aa1d-92b648cf0e58:image.png)
    

![image.png](attachment:b543ca35-48c0-488d-935e-2c76a6233884:image.png)

3. 파이썬 코드를 통한 질감 후처리
