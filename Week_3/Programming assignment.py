from google.colab import files
uploaded = files.upload()   # 會跳出上傳視窗，選擇 O-A0038-003.xml
print("已上傳檔案：", list(uploaded.keys()))
