from imageai.Classification import ImageClassification
import os
# below is the execution path which is the cmd pwd
execution_path = os.getcwd()

# instantiate the image prediction
prediction = ImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(execution_path, "mobilenet_v2.h5"))
prediction.loadModel()


# make a prediction
predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "godzilla.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)