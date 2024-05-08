function getFolderId(folderUrl) { //取得存放圖片的雲端資料夾的ID
  var regex = /folders\/([a-zA-Z0-9_-]+)/;
  var match = folderUrl.match(regex);
  return match ? match[1] : null;
}

function storeAsCSV(imageID) {
  var csvContent = 'image_name,image_ID\n';
  for (var i = 0; i < imageID.length; i++) {
    csvContent += imageID[i].join(",") + "\n";
  }

  csvfolderId = getFolderId('https://drive.google.com/drive/folders/1TG_hPx2K5H5zWxXfxjUCvKjS-71N_1ZX?usp=sharing');
  var csvFolder = DriveApp.getFolderById(csvfolderId);
  var csvFile = csvFolder.createFile("small_mammals_imageID.csv", csvContent);
}

function getImageUrl(folderUrl) { //取得雲端資料夾內圖片的ID
  var folderId = getFolderId(folderUrl);
  var folder = DriveApp.getFolderById(folderId);
  var files = folder.getFiles();
  var folderID =[];
  var imageID =[];
  for (var i = 0; files.hasNext(); i++){
  // for (var i = 0; i<10; i++){
      var folder = files.next();
      folderID[i] = folder.getId();
      var data =[folder.getName(),folder.getId()];
      imageID[i] = data;
  }
  imageID.sort(function() {return (0.5-Math.random());});
  storeAsCSV(imageID);
  return imageID
}

function makeForm(title, desc, folderUrl, option) { //做表單問卷
  var questionsPerPage = 20; //每頁問題數
  var questionsPerForm = 100; //每份問卷問題數
  
  var imageID = getImageUrl(folderUrl); //取得圖片的ID
  console.log(imageID[0][0])

  for (var i=0; i<imageID.length; i++) { // 製作每題題目和選項
    // 每100題一份問卷
    if (i % questionsPerForm === 0) {
      //設定表單的標題和說明文字，並讓使用者填寫姓名
      form = FormApp.create(title + '(' + (i/questionsPerForm+1) + ')');
      form.setDescription(desc);
      var nameItem = form.addTextItem();
      nameItem.setTitle('請輸入您的姓名:').setRequired(true);
    }

    // 放上圖片
    var googleUrl = 'https://drive.usercontent.google.com/u/3/uc?id=';
    var imgUrl = googleUrl + imageID[i][1] + '&export=download';
    var img = UrlFetchApp.fetch(imgUrl);
    Utilities.sleep(100);
    var imgItem = form.addImageItem();
    try {
      imgItem.setImage(img);
    } catch (e) {
      Utilities.sleep(500);
      imgItem.setImage(img);
    }
    
    // 放上圖片和選項，並將問題設為必須回答
    Utilities.sleep(10);
    var item = form.addMultipleChoiceItem();
    item.setTitle('Q' + (i+1) + ': 上圖是什麼動物?')
    .setChoices([
      item.createChoice(option[0]),
      item.createChoice(option[1]),
      item.createChoice(option[2]),
      item.createChoice(option[3]),
      item.createChoice(option[4])
    ])
    .setRequired(true)

    // 每20題換頁一次
    if ((i+1) % questionsPerPage === 0 && (i+1) < imageID.length) {
      form.addPageBreakItem();
    }
  }
}

function main() {
  var title = 'Cifar100的小型哺乳類動物(small mammals)問卷';
  var desc = 'Cifar100的小型哺乳類動物(small mammals)，包含倉鼠(hamster), 老鼠(mouse), 兔子(rabbit), 錢鼠(shrew), 松鼠(squirrel)，共五種類別，請選出每張圖片的動物為以上五種中的哪個類別。\nCifar100的小型哺乳類動物(small mammals)問卷總題數共500題，由於題數較多，因此問卷會分成5份，每份100題，請確定每份皆有填寫。';
  var folderUrl = 'https://drive.google.com/drive/folders/1sG6hAy3wlpOJkb5SuvUk2k17mFS-Cas-?usp=sharing';//圖片雲端資料夾地址
  var option = ['倉鼠', '老鼠', '兔子', '錢鼠', '松鼠'];
  makeForm(title, desc, folderUrl, option)
}