# Collect UPLL dataset details
* 共有兩份問卷，分別為small mammals小型哺乳類問卷、trees樹問卷
* 每份問卷皆有500題單選題，每題會給一張隨機的圖片及五個選項，標註者將選出他認為這張圖會是哪個物種的圖片
* 為了方便標註者填寫，每份問卷被拆成5個小問卷，每個小問卷皆有100題
* 標註者們問卷填寫的時間為(2024/2/21-2024/3/6)

# Automatically create Google forms
1. 請先在您的Google雲端中建立一個名為`create_forms`的資料夾，並將此資料夾中所有的檔案放入雲端中的`create_forms`資料夾中
2. 開始製作問卷前，先使用`randomly_select_images.ipynb`，為各類別取得隨機的100張圖片
    * 請依序執行此程式
    * 成功執行後，`create_forms`底下將會出現一個名為`select_images`的資料夾，內包含`cifar100_SM_T.zip`、`small_mammals.csv`、`trees.csv`、資料夾`small_mammals`、資料夾`trees`
    * 資料夾`/select_images/small_mammals/`及資料夾`/select_images/trees/`中各自有隨機選取的500張小型哺乳類、樹的圖片
    * 圖片上傳至雲端資料夾需要一點時間，請耐心等候
3. 在資料夾`create_forms`中建立一個名為`forms`的Google試算表
    * 開啟`forms`後，選擇工具列中的"擴充功能"中的"Apps Script"打開Google Apps Script
    * 將`create_small_mammals_forms.gs`或`create_trees_forms.gs`中的所有內容複製貼上至`程式碼.gs`並執行
    * 請分開執行`create_small_mammals_forms.gs`及`create_trees_forms.gs`，一次執行一個
    * 執行前，記得將工具列中的"要執行的程式"切換到"mainSM"及"mainT"
    * 執行前，記得修改參數folderUrl，修改為你自己Google雲端中資料夾`create_forms`的分享連結
    * 記得要把資料夾`create_forms`的分享權限改為"知道連結的任何人"
    * 成功執行後，`create_forms`底下將會出現一個名為`forms`的資料夾，內包含`small_mammals_imageID.csv`、`trees_imageID.csv`，另外你將在你的雲端主頁中得到5份small_mammals的問卷及5份trees的問卷，請手動將所有問卷移至`/create_forms/forms/`中
4. 最後您可以使用這10份問卷收集真人標註的UPLL資料
5. 標註收集結束後，整理問卷的code在此檔案的上一層資料夾`CLIG_FAPT_FATEL/dataset_collection/create_forms`中
