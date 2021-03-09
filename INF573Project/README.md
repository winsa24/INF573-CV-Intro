# INF573-Project

Transfer Yolov5 + Deep Sort with PyTorch (https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch) to colab  

Also get some ideas from https://github.com/ultralytics/yolov5 
(some chinese tutorials may help:https://zhuanlan.zhihu.com/p/156835045. || https://www.paddlepaddle.org.cn/tutorials/projectdetail/982177#anchor-33  ||https://github.com/xingkongliang/Pedestrian-Detection. )

** Colab Link : ** 
https://colab.research.google.com/drive/11uBBY5zZfpV5owzTNOqIOyJsgNQytm4j?usp=sharing

work tracker:
******
07/11/2020:  
  1. Finished environment transfer. (not in a good way:(  
  2. Test a video based on http, but failed  
******
15/11/2020:
  1. Reading <<YOLOv4: Optimal Speed and Accuracy of Object Detection>> --- Bag of freebies
  2. Draw an mind mapping to help understand the structure and each mentioned strategies
   ![image](https://github.com/winsa24/INF573-Project/blob/main/freebies(1).png)
******  
16/11/2020:
  1. Continue to read <<YOLOv4: Optimal Speed and Accuracy of Object Detection>> --- Bag of freebies  
  2. Complete mind mapping  
   ![image](https://github.com/winsa24/INF573-Project/blob/main/freebies(2).png)  
******   
18/11/2020:  
  1. Try to get the boundbox coordinate as a preparation to apply bokeh-effect.  
  YOLOv3: Cost a long time to load.. didn't try to test yet...  
    Tutorial:https://github.com/eriklindernoren/PyTorch-YOLOv3  
    Colab :https://colab.research.google.com/drive/1bLVh1y-ztPpPiO98qFuGaBsJIN7Zq2j2#scrollTo=LOYgqLYlsKYC      
  
  YOLOv4: Shell command. Can't use the return result directly....  
    Tutorial:https://colab.research.google.com/drive/12QusaaRj_lUwCGDvQNfICpa7kA7_a2dE#scrollTo=jESyRhd4Nd38  
    Colab link: https://colab.research.google.com/drive/1lCbwRTnjtPVXc1S5D0saTD4RGmUfGxRs#scrollTo=LO5Y40hvK9sy  
  
  YOLOv5: Works!  
    Tutorial : https://github.com/ultralytics/yolov5/issues/36  
    Colab link : https://colab.research.google.com/drive/1pgqgT-xQ3GQHlwCZYWR5ZNmi7pcHOb_A#scrollTo=vJre1aO3wN8X  
    Pytorch Hub: https://pytorch.org/hub/research-models  
******  
21/11/2020:  
  Colab link : https://colab.research.google.com/drive/1pgqgT-xQ3GQHlwCZYWR5ZNmi7pcHOb_A#scrollTo=ElBPcCBYqlC7    
  1. test my own picture  
  2. applied blur effect  
  Results as follow:  
  ![image](https://github.com/winsa24/INF573-Project/blob/main/results0.jpg)  
  ![image](https://github.com/winsa24/INF573-Project/blob/main/blured.jpg)  
******
28/11/2020:
  1.Try MaskRCNN, get error on training.   
  https://colab.research.google.com/drive/1u18PHiM-M7HoujRuJVhNRie_dBgotq9P#scrollTo=KacLbJF5_qnm    
  think of using YOLO substitute FastRCNN...    
  
  2.Try to find YOLO segmentation:    
  https://github.com/ArtyZe/yolo_segmentation    
  https://github.com/zhengshoujian/darknet-yolo-segmentation.   
  https://github.com/dbolya/yolact.   
  implementing...    
  useful link:  
  https://www.raywenderlich.com/books/machine-learning-by-tutorials/v2.0/chapters/10-yolo-semantic-segmentation#toc-chapter-013-anchor-002.  
  https://medium.com/@upchen_/%E5%A6%82%E4%BD%95%E5%9C%A8-colab-%E5%AE%89%E8%A3%9D-darknet-%E6%A1%86%E6%9E%B6%E8%A8%93%E7%B7%B4-yolo-v3-%E7%89%A9%E4%BB%B6%E8%BE%A8%E8%AD%98%E4%B8%A6%E4%B8%94%E6%9C%80%E4%BD%B3%E5%8C%96-colab-%E7%9A%84%E8%A8%93%E7%B7%B4%E6%B5%81%E7%A8%8B-e5ded7bbab00.  
  
  3.Try to calculate IoU of Bbox on previous YOLO result.   
  https://colab.research.google.com/drive/1pgqgT-xQ3GQHlwCZYWR5ZNmi7pcHOb_A#scrollTo=xQr3CTYIOUX1&line=9&uniqifier=1.    
  
  4.Search for mask IoU calculate method.    
  https://colab.research.google.com/drive/1u18PHiM-M7HoujRuJVhNRie_dBgotq9P#scrollTo=zUn3WzfcjbOf&line=1&uniqifier=1.   
  
  5.Search for edge smoothing.  
  https://towardsdatascience.com/smoothing-semantic-segmentation-edges-8b9240052904  
******
02/12/2020:   
  Discuss on *Google meeting* and start writting report!  
  https://www.overleaf.com/project/5fc63c8200533b034a285af8  
  Code tracing:
  https://colab.research.google.com/drive/1wnuluJ8vGARsm1OgNfyOKNXW1e0OoGzY?usp=sharing#scrollTo=0qwae8lBvXJ8.   (Thx Duo)
*****  
12/12/2020:  
  Task on my colab:  
  https://colab.research.google.com/drive/154qnch62faUZGiKoJ-XdROPi0P35Kc_Z#scrollTo=5QtQwGS_YqVU.  
  Working on report and presentation...  
  
*****  
16/12/2020:    
  Final code: https://github.com/thinhngo-x/portrait-mode   
  Final slide: https://docs.google.com/presentation/d/13qDTnocyCO-xc3R9xh8sw_BdX4v_Yh5sykxQRlEWjts/edit?usp=sharing. 
  Final report: https://www.overleaf.com/project/5fd8b5c0843d3c718e1f24bb
