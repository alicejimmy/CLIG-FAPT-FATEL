# Automatically create Google forms
## 程式使用步驟
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
4. 之後您將可以使用這10份問卷收集真人標註的UPLL資料
5. 標註收集結束後，可以使用`organize_forms_result.ipynb`整理問卷結果
    * 執行前，記得修改參數true_label_ID, form_order_ID, form_result_IDs，修改為你自己Google雲端中各檔案的ID
    * 成功執行後，`create_forms`中將會出現`result.csv`

## 檔案介紹
所有程式執行結束後，雲端中應該有以下檔案
### `create_forms/`中
* `randomly_select_images.ipynb`：為每類別隨機選取100張圖片，並儲存至雲端
* `forms`：Google試算表，用於自動製作樹的問卷
   * `forms`的Apps Script中`create_small_mammals_forms.gs`：自動製作小型哺乳類的問卷
   * `forms`的Apps Script中`create_trees_forms.gs`：自動製作樹的問卷
* `organize_forms_result.ipynb`：整理問卷調查結果
* `result.csv`：small_mammals或trees的問卷調查結果整理
### `create_forms/select_images/`中
* `cifar100_SM_T.zip`：2500張小型哺乳類的圖片及2500張樹的圖片
* `small_mammals.csv`：儲存隨機挑選500張小型哺乳類的檔名及正確類別
* `trees.csv`：儲存隨機挑選500張樹的檔名及正確類別
* `small_mammals`：隨機挑選的500張小型哺乳類圖片
* `trees`：隨機挑選的500張樹圖片
### `create_forms/forms/`中
* `small_mammals_imageID.csv`：儲存問卷中隨機排序後500張小型哺乳類圖片的順序
* `trees_imageID.csv`：儲存問卷中隨機排序後500張樹圖片的順序
* 5份小型哺乳類的Google問卷
* 5份樹的Google問卷
## 補充說明
### Google雲端中檔案的ID
* 在Google雲端點選某個檔案將會進入預覽檔案模式，此時URL中的ID為目前資料夾的ID並非該檔案的ID
* 若要查看該檔案的ID，請點選"在新視窗中開啟"，此時的URL中的ID為該檔案的ID
   * 例如檔案：https://drive.google.com/file/d/1AemdxSabDnYd_XbVMcgDFeOqOe9ZG4nP/view，其ID為1AemdxSabDnYd_XbVMcgDFeOqOe9ZG4nP
![image](https://github.com/alicejimmy/CLIG_FAPT_FATEL/assets/71706978/da052018-ccec-4d58-a2cb-f8a15594b3d3)
![image](https://github.com/alicejimmy/CLIG_FAPT_FATEL/assets/71706978/02f72d9d-eae4-49de-913d-36a54cfc38f7)


