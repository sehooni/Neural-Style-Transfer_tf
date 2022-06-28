# Neural Style Transfer-tf
opencv 및 tf.keras를 활용한 Neural Style Transfer 프로젝트 및 개념, 예제 정리 repo

## Project
### Environment Setting
- python 3.6.5  
- Cuda 11.1 / tensorflow-gpu 2.4.0

### Detail of Project
![프로젝트 설명 슬라이드 1](https://user-images.githubusercontent.com/84653623/165943832-b93feebd-beb6-4965-970b-360836194f44.png)

![프로젝트 설명 슬라이드 2](https://user-images.githubusercontent.com/84653623/165943884-6d3c0559-dec9-4d59-8ec0-184fef17b99b.png)

### Purpose of Project
1. 논문 구현을 하면서 제작한 [Jupyter Notebook file](https://github.com/sehooni/Neural-Style-Transfer_tf/blob/master/project/Neural_Style_Transfer_code.ipynb)
을 [python file](https://github.com/sehooni/Neural-Style-Transfer_tf/blob/master/project/Neural_Style_Transfer.py) 형식으로 바꾸면서 코드를 조금더 간결화 한다.
2. 이후 실시간 영상에 적용할 수 있는 코드 작성한다. 다양한 예제들이 존재하지만 구성 코드와 환경, 목적에 따라 다르게 작동하므로 나만의 코드를 작성한다.
3. 현재 논문에서 적용된 딥러닝 모델은 VGG19이다. 이후 다양한 모델들이 등장했는데, 시간적 여유가 존재한다면, 또 다른 딥러닝 모델을 적용하여 그 차이점 및 장단점을 확인한다.

![프로젝트 설명 슬라이드 3](https://user-images.githubusercontent.com/84653623/165943965-d9363c97-f501-458f-b52e-7aeeb7b2c7d4.png)


## opencv 예제
[OpenCV 4로 배우는 컴퓨터 비전과 머신 러닝(황선규)](https://github.com/sehooni/Neural-Style-Transfer_tf/blob/master/opencv%20%EC%98%88%EC%A0%9C/README.md)
에 포함된 python을 이용한 opencv예제이다.


## Reference
본 repository는 Opencv 기법 및 tf.keras가 적용된 'Neural-Style-Transfer' project이다.


### PAPER
- [A Neural Algorithm of Artistic Style.2015](https://arxiv.org/abs/1508.06576) 에서 제시된 Mechanism을 이용하여 본 프로젝트를 진행한다.

### OPENCV 참고교재
- OpenCV 4로 배우는 컴퓨터 비전과 머신 러닝(황선규)
  - [깃허브링크](https://github.com/sunkyoo/opencv4cvml)
- OpenCV-Python으로 배우는 영상처리 및 응용


----
# 수정사항
(22.06.28) [구성 tree 및 results upload 관련 issue](https://github.com/sehooni/Neural-Style-Transfer_tf/blob/master/project/data.md)
