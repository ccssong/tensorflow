import numpy as np
import librosa
import glob


row = 48
feature_num = 39

def extract_feature(file_name):
   print('extract')
   X, sample_rate = librosa.load(file_name, duration=1.1)
   mfccs = librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc = feature_num).T
   print('mfcc',mfccs.shape)
   print('row: ',row)
   return mfccs


def parse_audio_files(filename):
   ans = [0, 0]
   ans[1] = 1
   print('parse',filename)
   features = np.zeros((row, feature_num))
   features = extract_feature(filename)
   #print('features: \n', features)
   return features, ans


files = []
files = glob.glob('./안녕하세요/*.wav')
length = len(files)
print('length: ',length)
for i in range(length):
   eachfile = './안녕하세요/'+str(i)+'.wav'
   X, Y = parse_audio_files(eachfile)
   npzFileName = './안녕하세요/Array/'+str(i)
   np.savez(npzFileName, X=X, Y=Y)




""""i = 0
file = './안녕하세요/' + str(i) + '.wav'
parse_audio_files(file)

i = 1
file = './안녕하세요/' + str(i) + '.wav'
parse_audio_files(file)



feature_num = 39

tc = []
td = []
ans = [0, 0]
ans[1] = 1

X, sample_rate = librosa.load('./안녕하세요/0.wav', duration=1.1)
mfcc1 = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=feature_num).T
print('shape1: ', mfcc1.shape , '\n' ,mfcc1)

X, sample_rate = librosa.load('./안녕하세요/1.wav', duration=1.1)
mfcc2 = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=feature_num).T
print('shape2: ', mfcc2.shape , '\n' ,mfcc2)

total = []
total = np.append(mfcc2, mfcc1)
print('\n\ntotal: ', total.shape)

for i in range(14):
   file_name = './안녕하세요/' + str(i) + '.wav'
   print(file_name)
   X, sample_rate = librosa.load(file_name, duration=1.1)
   mfccs = librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc=feature_num).T
   #print(mfccs.shape)

   for col in range(feature_num):
      narr = []
      s_n = mfccs[:, col]
      tc.append(s_n)
      td.append(ans)
      #print('tc\n')
      #print(tc)
   print(tc)

tc = np.array(tc)
td = np.array(td)
npz_file = './안녕하세요/Array/mytotaldata'
np.savez(npz_file, X = tc, Y = td)

print('tc\n')
#print(tc)
print(tc.shape)"""
