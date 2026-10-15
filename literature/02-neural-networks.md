# 2. Нейросети (до LLM)

Параметры отвечают за память, FLOPs за время. Для полносвязного слоя `F ≈ 2P`, свёртка эту пропорцию ломает. Активации считаются как пик по расписанию, а не как сумма по слоям. 
До 2020 года инференс CNN был ограничен вычислениями, а обучение уже упиралось в память.

## Период, когда случился переход к обучению на нескольких GPU

- Krizhevsky, Sutskever, Hinton. *ImageNet Classification with Deep Convolutional Neural Networks (AlexNet).* NeurIPS 2012. <https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>
- Simonyan, Zisserman. *Very Deep Convolutional Networks (VGG).* ICLR 2015. <https://arxiv.org/abs/1409.1556>
- He et al. *Deep Residual Learning (ResNet).* CVPR 2016. <https://arxiv.org/abs/1512.03385>
- Sutskever, Vinyals, Le. *Sequence to Sequence Learning.* NeurIPS 2014. <https://arxiv.org/abs/1409.3215>

## Параметры, FLOPs и арифметика слоёв

- Raschka. *Machine Learning Q and AI*, гл. 11. <https://sebastianraschka.com/books/ml-q-and-ai-chapters/ch11/>
- Carreira-Perpiñán et al. *Dissecting FLOPs along input dimensions for GreenAI cost estimations.* 2021. <https://arxiv.org/abs/2107.11949>
- Sze, Chen, Yang, Emer. *Efficient Processing of Deep Neural Networks: A Tutorial and Survey.* Proc. IEEE, 2017. <https://arxiv.org/abs/1703.09039>
- Canziani, Paszke, Culurciello. *An Analysis of Deep Neural Network Models for Practical Applications.* 2016. <https://arxiv.org/abs/1605.07678>
- Han, Mao, Dally. *Deep Compression.* ICLR 2016. <https://arxiv.org/abs/1510.00149>
- Micikevicius et al. *Mixed Precision Training.* ICLR 2018. <https://arxiv.org/abs/1710.03740>
- Rajbhandari et al. *ZeRO.* SC'20. <https://arxiv.org/abs/1910.02054>

## Аналитические подходы vs профилирование

- Gao et al. *Estimating GPU Memory Consumption of Deep Learning Models (DNNMem).* ESEC/FSE 2020. <https://www.microsoft.com/en-us/research/wp-content/uploads/2020/09/dnnmem.pdf>
- Yousefzadeh-Asl-Miandoab et al. *GPU Memory and Utilization Estimation for Training-Aware Resource Management.* 2026. <https://arxiv.org/abs/2602.17817>
- Sohoni et al. *Low-Memory Neural Network Training.* 2022. <https://arxiv.org/abs/1904.10631>
