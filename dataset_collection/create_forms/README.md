# create Google forms
1. 請先在您的Google雲端中建立一個名為"create_forms"的資料夾，並將此資料夾中所有的檔案放入雲端中的"create_forms"資料夾中
2. 開始製作問卷前，先使用randomly_select_images.ipynb，為各類別取得隨機的100張圖片
    * 請依序執行此程式
    * 成功執行後，"create_forms"底下將會出現一個名為"select_images"的資料夾，內包含cifar100_SM_T.zip、small_mammals.csv、trees.csv、資料夾"small_mammals"、資料夾"trees"
    * 資料夾"/select_images/small_mammals/"及資料夾"/select_images/trees/"中各自有隨機選取的500張小型哺乳類、樹的圖片
    * 圖片上傳至雲端資料夾需要一點時間，請耐心等候
3. 在資料夾"create_forms"中建立一個名為"create_forms"的Google試算表
    * 開啟create_forms後，選擇工具列中的"擴充功能"中的"Apps Script"打開Google Apps Script
    * 將create_small_mammals_forms.gs或create_trees_forms.gs中的所有內容複製貼上至"程式碼.gs"
    * 執行前，記得將create_small_mammals_forms.gs及create_trees_forms.gs工具列中的"要執行的程式"切換到"mainSM"及"mainT"
    * 執行前，記得修改create_small_mammals_forms.gs及create_trees_forms.gs中的參數folderUrl，修改為你自己Google雲端中資料夾"create_forms"的分享連結
    * 記得要把資料夾"create_forms"的分享權限改為"知道連結的任何人"
