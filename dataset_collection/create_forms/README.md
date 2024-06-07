# Automatically create Google forms
## Steps for usage
1. create a folder named `create_forms` in your Google Drive and place all the files from this page into the `create_forms` folder.
2. Before starting to create the forms, use `randomly_select_images.ipynb` to obtain random 100 images for each class.
   * Execute each cell sequentially.
   * Upon successful execution of all cells, the folder `create_forms/select_images` will appear.
      * It contains `cifar100_SM_T.zip`, `small_mammals.csv`, `trees.csv`, `small_mammals` folder, and `trees` folder.
         * The `small_mammals` and `trees` folders each contain randomly selected 500 images.
   * Uploading images to the cloud folder may take some time, so please be patient.
3. Create a Google Sheets named `forms` in the `create_forms` folder.
   * Open `forms`, and then open Google Apps Script.
      * Extensions -> Apps Script.
   * Copy and paste the content of `create_small_mammals_forms.gs` or `create_trees_forms.gs` into `code.gs` and run it.
      * Remember to run `create_small_mammals_forms.gs` and `create_trees_forms.gs` separately. One at a time.
      * Before execution, remember to switch to the "main" function.
      * Before execution, remember to modify the parameter "folderUrl" to the shareable link of your `create_forms` folder in your Google Drive.
         * Remember to change the sharing permission of the `create_forms` folder to "Anyone with the link".
   * Upon successful execution, the following will appear in your Google Drive:
      * `create_forms/forms/small_mammals_imageID.csv`
      * `create_forms/forms/trees_imageID.csv`
      * Five small mammal forms and five tree forms on the main page of your cloud storage.
         * Manually move all the forms to `create_forms/forms/`.
4. Now you can use these 10 forms to collect Human-annotated Labels of UPLL Dataset.
5. After finish collecting the labels, you can use `organize_forms_result.ipynb` to organize the results of forms.
   * Before execution, remember to modify the parameters `true_label_ID`, `form_order_ID`, `form_result_IDs` to the IDs of your own files.
   * Upon successful execution, `create_forms/result.csv` will appear.

## File Description
After all codes are executed, you will get the following files in Google Drive:

### In `create_forms/`
* `randomly_select_images.ipynb`: Randomly selects 100 images for each class and saves them to the Google Drive.
* `forms`: A Google Sheets used to automatically create forms.
   * `create_small_mammals_forms.gs`: Automatically creates forms for small mammals.
   * `create_trees_forms.gs`: Automatically creates forms for trees.
* `organize_forms_result.ipynb`: Organizes the results of forms.
* `result.csv`: Organized  results of forms for small mammals or trees.

### In `create_forms/select_images/`
* `cifar100_SM_T.zip`: 2500 images of small mammals and 2500 images of trees.
* `small_mammals.csv`: Randomly selected filenames and their ground-truth labels for 500 small mammal images.
* `trees.csv`: Randomly selected filenames and their ground-truth labels for 500 tree images.
* `small_mammals`: Stores randomly selected 500 images of small mammals.
* `trees`: Stores randomly selected 500 images of trees.

### In `create_forms/forms/`
* `small_mammals_imageID.csv`: The order of 500 randomly arranged small mammal images in the form.
* `trees_imageID.csv`: The order of 500 randomly arranged tree images in the form.
* Five Google froms for small mammals.
* Five Google forms for trees.

## Supplementary information
### IDs of files in Google Drive
* Clicking on a file in Google Drive will enter the preview mode of this file. The ID in the URL at this time is the ID of this current folder, not the ID of this file.
* To view the ID of this file, please click "Open in new window". The ID in the URL at this time is the ID of this file.<br>
  For example: <br>
  The URL of a file: https://drive.google.com/file/d/1AemdxSabDnYd_XbVMcgDFeOqOe9ZG4nP/view <br>
  The ID of this file: 1AemdxSabDnYd_XbVMcgDFeOqOe9ZG4nP

![image](https://github.com/alicejimmy/CLIG-FAPT-FATEL/assets/71706978/7940e5bc-30e9-4cb4-9727-9a655f973e72)
