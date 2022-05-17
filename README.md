# Human-Eye-Diagnosis-for-Cataract-Detection


Cataracts are one of the most common visual problems and it affects people as they grow older and lose their vision. A cataract is a cloud that forms on the lens of our eyes and it is caused by a buildup of debris. Among the most noticeable signs of this illness are blurred vision, faded colors and difficulties seeing in bright light. It is common for these symptoms to result in trouble doing a number of duties. Therefore, early cataract detection and prevention may aid in reducing the rate of blindness in the population. On the basis of a publicly available image dataset, we hope to recognize cataract eye disease using convolutional neural networks. As part of this experiment, six alternative Convolutional Neural Network (CNN) meta-architectures, including NetInceptionV3, XceptionNet,MobileNet, EfficientNetV2B1, EfficientNetV2B0 and DenseNet121 were applied to the TensorFlow object detection framework, with each architecture being represented by a different color.


      1.Problem Definition (Problem Statement)
      2.Data Collection
      3.Data pre-processing
      4.Modeling with CNN(Transfer Learning)
      5.Evaluating
      

# Problem Statement
In the event that Cataract is not detected and treated in its early stages, it has the potential to cause full blindness of the eye. Based on the information our objective is to develop a categorization model that may assist in determining whether or not a person had Cataract. The pictures acquired by the Fundus camera are used as input for the model.


# Data Explore & Visualization

Dataset collected from kaggle 
 https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k?fbclid=IwAR1Z_QCreIQhuHic-CEWJU4r9TEg_rcmCkkN-0QqyNMzom38vRtzZbUHPx4
 
 
Visualizing retina funduscopy images

![Screenshot 2022-05-17 205557](https://user-images.githubusercontent.com/54286216/168842481-83d11bd3-1b14-4154-8206-09f065ba384d.png)
Funduscopic images of left and right eyes with labeling




![2](https://user-images.githubusercontent.com/54286216/168843271-21ddadbe-daaf-4e23-a966-68f23bcb00af.png)
 A tabular dataset or csv file record which holds the target or label of each victim's retina image accordingly to image folder index



# Data Pre-processing

   Extracted all the fundus images that carries cataract and conventional fundus photographs, into two phases: the first phase that labels 'N' and the second phase that labels 'C'. Data was filtered using labels, which were assigned to each row. Because they were taken using different cameras, the image sizes of the experimental fundus photos were not consistent. As a consequence, we used OpenCV to resize the image to 224*224 pixels, which was a good fit. The dataset is then loaded and translated into an array format for use in training using the NumPy library, which is then used to store the result.
   
  
  ![3](https://user-images.githubusercontent.com/54286216/168846933-3c54cd17-a40e-4519-a230-3fa2fb28c1d7.png)
  Total Number of images to be used after processing
  
 


# Modelling

Transfer Learning approach(pretrained CNN models)

    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3),
    activation='relu'
    GlobalAveragePooling2D(),
    BatchNormalization(),
    activation='sigmoid'



      optimizer = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
      loss="binary_crossentropy"
      batch_size=32,
      epochs=15
              
      
      
# Results & Evaluation

Models performance values

![Screenshot 2022-05-17 231819](https://user-images.githubusercontent.com/54286216/168874341-a4b7de4a-565c-49b2-a613-8c6116ef88a9.png)

Accuracy chart

![Screenshot 2022-05-17 232359](https://user-images.githubusercontent.com/54286216/168874438-69d663bc-d315-4615-a6c3-dc239065a2de.png)

Loss chart

![Screenshot 2022-05-17 232715](https://user-images.githubusercontent.com/54286216/168874483-1ce556d6-8309-4690-abba-51d375628357.png)


EfficientNetV2B0 has the highest number of accuracy. Although it's Loss value very slightly greater than MobileNet, but it can be neglected. However, MobileNet's accuracy performance is pretty good but not fine tuned like EfficientNetV2B0. We can come to conclusion that EfficientNetV2B0 is the best model. EfficientNetV2B1 models accuracy performance is similar with MobileNet, but Loss value is twice of it.





