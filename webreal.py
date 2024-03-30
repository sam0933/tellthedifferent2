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




def make(colornumber1,imagess,labelss,many,s,ph):
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
    X_train = X_train.reshape(-1,ph,ph, colornumber1)
    X_test = X_test.reshape(-1,ph,ph, colornumber1)
    #在你的模型中，input_shape 設置為 (28, 28, 1)，這意味著每個輸入圖像的形狀應該是 (28, 28, 1)，而不是 (28, 28, 1)。這是因為模型期望接收單個圖像的形狀，而不是整個訓練集的形狀。所以在訓練模型時，你不需要指定樣本數，只需要指定圖像的高度、寬度和通道數。
    model = Sequential([Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(ph,ph,colornumber1))])
    if len(y_train)<=100:

        model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(many, activation='softmax'))
    elif len(y_train)<=500:
        model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(many, activation='softmax'))
    elif len(y_train)<=2000:
        model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(many, activation='softmax'))
    elif len(y_train)<=5000:
        model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(many, activation='softmax'))
    else:
        model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(many, activation='softmax'))



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
        st.write('1選擇要分的「種類數」、「用黑白或彩色」、「畫質」')
        st.write('2輸入每一類的名稱及上傳各類的訓練資料')
        st.write('3按下「開始訓練」\n4在顯示訓練完成後按下「下載準備」再按「下載模型文件」')
        st.write('4或是在下方使用模型的地方上傳要辨識的圖片接著最下方就會有一串列顯示第一張(第0項)是什麼類第二張是什麼(可上傳多張)')
    cate = st.slider('請選擇要分的類別數', min_value=1, max_value=10, value=2)
    color = st.selectbox('請選擇要用「黑白」或「彩色」',('grayscale','rgb'))
    if color=='grayscale':
        colornumber=1
    else:
        colornumber=3
    ph1 = st.slider('請選擇畫質', min_value=1, max_value=500, value=100)
    a=0
    lab123=[]
    for i in range(cate):
        lab = st.text_input(f"label {i+1}:")
        imgs = st.file_uploader(f"上傳第 {i+1} 類圖片", type=["jpg", "png"], accept_multiple_files=True)
        for img in imgs:
            img_pil = image.load_img(img, target_size=(ph1,ph1), color_mode=color)
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
            st.write("訓練已開始，請不要做任何操作以避免訓練中斷")
            trained_model = make(colornumber,all_images,all_labels,cate,z,ph1)
            trained_model.save('model.h5')
            trained_model.save('model123.keras')
    if os.path.exists('model.h5') or os.path.exists('model123.keras'):
        if st.button('刪除模型重新建立'):
            for i in range(10):
                if os.path.exists('model123.keras'):
                    os.remove('model123.keras')
                if os.path.exists('model.h5'):
                    os.remove('model.h5')
            st.success('已刪除')
            st.info('請再操作一遍，建立新模型')
    
    if os.path.exists('model.h5'):
        with open('model.h5','rb') as model_file:
            model_binary = model_file.read()
        download_button = st.download_button(label="下載模型文件", data=model_binary, file_name="trained_model.h5", mime="application/octet-stream")
        if download_button:
            st.success("文件已開始下載")
    result=[]

    if os.path.exists('model123.keras'):
        imgs1 = st.file_uploader("使用模型", type=["jpg", "png"], accept_multiple_files=True)
    #if imgs1:  是當imgs1不為空串列時==True
        if imgs1:
            model=load_model('model123.keras')
        for img in imgs1:
            img_pil = image.load_img(img, target_size=(ph1,ph1), color_mode=color)
            img_array = image.img_to_array(img_pil)
            v=img_array.reshape(1,ph1,ph1,colornumber)
            v =v.astype('float32') / 255.0
            v = model.predict(v)
            result.append(lab123[(np.argmax(v))])
        #if result:  是當resu;t不為串列時==True
        if result:
            st.write(result)

    code1='''
            import numpy as np
            from keras.preprocessing import image
            from tensorflow.keras.models import load_model
            model = load_model('你下載的模型的路徑(要含副檔名)')
            v=image.load_img('你想辨識的照片的路徑(要含副檔名)',target_size=(你當時選的畫質,你當時選的畫質), color_mode='grayscale'或'rgb')#grayscale是黑白  rgb是彩色  

            label=['你的第一類的名稱','你的第二類的名稱','你的第n類名稱']
            v=image.img_to_array(v)

            #是四維陣列要說第一個1是總比數一定要寫
            v=v.reshape(1,你當時選的畫質,你當時選的畫質,1或3)#黑白用1   彩色用3


            v =v.astype('float32') / 255.0

            v = model.predict(v)
            print(label[np.argmax(v)])
            '''
    st.write('下面是在電腦使用模型的程式碼')
    st.warning('請確保您的電腦環境中以裝有streamlit、numpy、keras、scikit-learn、tensorflow')
    st.code(code1, language='python')


if __name__ == "__main__":
    main()
