from deepface import DeepFace

img1_path = "Resources/Images/852741.png"
img2_path = "Resources/Images/963852.png"

# Verify faces
result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path)
print("Is verified: ", result["verified"])

# Analyze emotion
emotion_result = DeepFace.analyze(img_path=img1_path, actions=['emotion'])
emotion_prediction = emotion_result[0]['emotion']
print("Emotion prediction: ", emotion_prediction)

