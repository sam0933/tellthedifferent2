import streamlit as st
import numpy as np
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
import streamlit as st
from keras.metrics import Accuracy
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score, f1_score
import time
import keras
import os
from tensorflow.keras.models import load_model




def make(imagess,labelss,many,s,ph):
    X = np.array(imagess)
    y = np.array(labelss)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/10, random_state=42)
    y_tts=y_test.tolist()
#plt.imshow(X_train[100])

# 將標籤轉換為 one-hot 編碼
    y_train = to_categorical(y_train, num_classes=many)
    y_test = to_categorical(y_test, num_classes=many)

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

#plt.imshow(X_train[40])
#當你將 X_train 變形為 (-1, 28, 28, 1) 時,-1 的意思是自動計算樣本數
    X_train = X_train.reshape(-1,ph,ph, 1)
    X_test = X_test.reshape(-1,ph,ph, 1)
    model = Sequential([
    #在你的模型中，input_shape 設置為 (28, 28, 1)，這意味著每個輸入圖像的形狀應該是 (28, 28, 1)，而不是 (28, 28, 1)。這是因為模型期望接收單個圖像的形狀，而不是整個訓練集的形狀。所以在訓練模型時，你不需要指定樣本數，只需要指定圖像的高度、寬度和通道數。
        Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(ph,ph, 1)),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(many, activation='softmax')
    ])
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    fi=0
    for i in range(s):
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=64, callbacks=[checkpoint])
        fi=fi+1
        y_pred = model.predict(X_test)
        ytf=[]
        for i in y_pred:
            ytf.append(int(np.argmax(i)))
        print(ytf)
        g=0
        for i in range(len(ytf)):
            if ytf[i]==y_tts[i]:
                g=g+1
        st.write(f'目前模型準確度:{g/len(ytf)}')
        st.write(f'已完成:{fi/s}')
    st.success('模型建立完成')

    return model












def main():
    all_images=[]
    all_labels=[]
    st.title("自動化影像辨識模型建立")
    if st.button('使用說明'):
        st.write('1選擇要分的種類數類和畫質')
        st.write('2輸入每一類的名稱及上傳各類的訓練資料')
        st.write('3按下「開始訓練」\n4在顯示訓練完成後按下「下載準備」再按「下載模型文件」')
        st.write('4或是在下方使用模型的地方上傳要辨識的圖片接著最下方就會有一串列顯示第一張(第0項)是什麼類第二張是什麼(可上傳多張)')
    cate = st.slider('請選擇要分的類別數', min_value=1, max_value=10, value=2)
    ph1 = st.slider('請選擇畫質', min_value=1, max_value=500, value=100)
    a=0
    lab123=[]
    for i in range(cate):
        lab = st.text_input(f"label {i+1}:")
        imgs = st.file_uploader(f"上傳第 {i+1} 類圖片", type=["jpg", "png"], accept_multiple_files=True)
        for img in imgs:
            img_pil = image.load_img(img, target_size=(ph1,ph1), color_mode='grayscale')
            img_array = image.img_to_array(img_pil)
            all_images.append(img_array)
            all_labels.append(a)
        lab123.append(lab)
        a=a+1
  

    if   len(lab)  <cate:
        st.warning("請上傳圖片以建立模型")
    else:
        z = st.slider('請選擇訓練次數', min_value=1, max_value=30, value=1)
        if st.button('開始訓練'):
            st.write("訓練已開始")
            trained_model = make(all_images,all_labels,cate,z,ph1)
            trained_model.save('model.h5')
            trained_model.save('model123.keras')
    
    if st.button("下載準備"):
        with open('model.h5','rb') as model_file:
            model_binary = model_file.read()
        os.remove('model.h5')
        download_button = st.download_button(label="下載模型文件", data=model_binary, file_name="trained_model.h5", mime="application/octet-stream")
        if download_button:
            st.write("文件下載中...")
    result=[]


    imgs1 = st.file_uploader("使用模型", type=["jpg", "png"], accept_multiple_files=True)
    #if imgs1:  是當imgs1不為空串列時==True
    if imgs1:
        model=load_model('model123.keras')
    for img in imgs1:
        img_pil = image.load_img(img, target_size=(ph1,ph1), color_mode='grayscale')
        img_array = image.img_to_array(img_pil)
        v=img_array.reshape(1,ph1,ph1,1)
        v =v.astype('float32') / 255.0
        v = model.predict(v)
        result.append(lab123[(np.argmax(v))])
        
    #if result:  是當resu;t不為串列時==True
    if result:
        st.write(result)


if __name__ == "__main__":
    main()
