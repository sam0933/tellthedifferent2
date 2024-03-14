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
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score, f1_score
import time
import keras



def make(imagess,labelss,many,s):
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
    X_train = X_train.reshape(-1,100, 100, 1)
    X_test = X_test.reshape(-1,100, 100, 1)
    model = Sequential([
    #在你的模型中，input_shape 設置為 (28, 28, 1)，這意味著每個輸入圖像的形狀應該是 (28, 28, 1)，而不是 (28, 28, 1)。這是因為模型期望接收單個圖像的形狀，而不是整個訓練集的形狀。所以在訓練模型時，你不需要指定樣本數，只需要指定圖像的高度、寬度和通道數。
        Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(100,100, 1)),
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
    checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
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



        # accuracy = Accuracy()
        # accuracy.update_state(y_test, y_pred)
        # result = accuracy.result().numpy()
        # st.write("Accuracy:", result)
        # # ac=accuracy_score(y_test, y_pred)
        # accuracy1 = model.evaluate(X_test, y_test)
        st.write(f'已完成:{fi/10}')
        # st.write(f'目前模型準確度{accuracy1}')
    model.save("test.h5")
    return model














def main():
    all_images=[]
    all_labels=[]
    st.title("自動化影像辨識模型建立")
    
    # 使用 Session State

    # 檢查 session state 中是否存在 x，如果不存在，則初始化為 1


    # 增加一個選擇框，讓用戶選擇 x 的值
    cate = st.slider('請選擇要分的類別數', min_value=1, max_value=10, value=1)
    a=0
    check=0
    v=[]
    # 創建 x 個標籤和文件上傳器
    for i in range(cate):
        lab = st.text_input(f"label {i+1}:")
        imgs = st.file_uploader(f"上傳第 {i+1} 種圖片", type=["jpg", "png"], accept_multiple_files=True)
        for img in imgs:
            img_pil = image.load_img(img, target_size=(100, 100), color_mode='grayscale')
            img_array = image.img_to_array(img_pil)
            all_images.append(img_array)
            all_labels.append(a)
            v.append(lab)  # 將類別添加到標籤列表中
        a=a+1
        tl=set(all_labels)

    if   len(lab)  <cate:
        st.warning("請上傳圖片以建立模型。")
    else:
        #st.write(check)
        z = st.slider('請選擇訓練次數', min_value=1, max_value=30, value=1)
        if st.button('開始訓練'):
            st.write("訓練已開始")
            make(all_images,all_labels,cate,z)
            st.write("模型建立完成")
        

if __name__ == "__main__":
    main()













