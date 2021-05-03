import torchaudio
from speechbrain.pretrained import EncoderClassifier
from scipy import spatial


class SpeakerVerifier():
  
  def __init__(self):
    self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", 
                                                savedir="pretrained_models/spkrec-xvect-voxceleb")
    self.enrolled_embeddings = []
    self.version = "0.0.1"
    self.enrolled_speaker = []


  def enroll(self, user_voice_path, speaker_name):
    signal, fs = torchaudio.load(user_voice_path)
    embedding = self.classifier.encode_batch(signal)
    self.enrolled_embeddings.append(embedding[0][0].numpy())
    self.enrolled_speaker.append(speaker_name)


  def verity(self, user_voice_path,name):
    signal, fs = torchaudio.load(user_voice_path)
    embedding = self.classifier.encode_batch(signal)[0][0].numpy()

    cosine_min = 100000
    for i,speaker in enumerate(self.enrolled_embeddings):
      cosine_distance = spatial.distance.cosine(embedding, speaker)

      if cosine_distance < cosine_min:
        cosine_min = cosine_distance
        detected_speaker = self.enrolled_speaker[i]

    return name == detected_speaker  
      
      
if __name__=="__main__":
    speaker_verifier = SpeakerVerifier()
    speaker_verifier.enroll("maryam_sv.wav", "maryam")
    speaker_verifier.enroll("mohammad_sv.wav", "mohammad")

    test1 = peaker_verifier.verity("maryam_sv.wav", "maryam")
    print(test1)
    test2 = speaker_verifier.verity("maryam_sv.wav", "mohammad")
    print(test2)
    test3 = speaker_verifier.verity("mohammad_sv.wav", "maryam")
    print(test3)
    test4 = speaker_verifier.verity("mohammad_sv.wav", "mohammad")
    print(test4)



