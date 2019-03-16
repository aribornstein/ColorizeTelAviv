let imgList = document.getElementById("imgList"),
  scrollRight = document.getElementById("scroll-right"),
  scrollLeft = document.getElementById("scroll-left");
// When a user clicks on the right arrow, the ul will scroll 750px to the right
scrollRight.addEventListener("click", event => {
  imgList.scrollBy(750, 0);
});

// When a user clicks on the left arrow, the ul will scroll 750px to the left
scrollLeft.addEventListener("click", event => {
  imgList.scrollBy(-750, 0);
});

async function main() {
  // tf.tidy(async () => {
  const feat_extractor = await tf.loadLayersModel(
    "https://cvworkshop.blob.core.windows.net/feat-extractor/model.json"
  );
  const model = await tf.loadLayersModel(
    "https://cvworkshop.blob.core.windows.net/tfjs-color/model.json"
  );
  fetch(
    "https://cvworkshop.blob.core.windows.net/telaviv-bw/?restype=container&comp=list"
  )
    .then(response => response.text())
    .then(str => new window.DOMParser().parseFromString(str, "text/xml"))
    .then(xml => {
      let blobList = Array.from(xml.querySelectorAll("Url")); //.getAttribute("Url");
      blobList.forEach(async entry => {
        let li_entry = document.createElement("li"),
          img = document.createElement("img");
        img.onload = async () => {
          let canvas = document.createElement("canvas");
          canvas.width = 224;
          canvas.height = 224;
          let img_tensor = tf.browser.fromPixels(img).toFloat(),
            rgb_img = tf.tidy(() => {
              return colorize(img_tensor, feat_extractor, model);
            });

          await tf.browser.toPixels(rgb_img.div(255), canvas);

          img.src = canvas.toDataURL();
          img.onload = null;
          // console.log(`${entry.title} is loaded colorize here`);
        };
        img.crossOrigin = "Anonymous";
        img.src = entry.innerHTML;
        li_entry.appendChild(img);
        imgList.appendChild(li_entry);
      });
      // });
    });
  // fetch("/src/telaviv.json")
  //   .then(response => {
  //     return response.json();
  //   })
  //   .then(function(myJson) {
  //     myJson.results.slice(0, 2).forEach(async entry => {
  //       let li_entry = document.createElement("li"),
  //         img = document.createElement("img");
  //       img.onload = async () => {
  //         var canvas = document.createElement("canvas");
  //         canvas.width = 224;
  //         canvas.height = 224;
  //         var img_tensor = tf.browser.fromPixels(img).toFloat();
  //         var rgb_img = colorize(img_tensor, feat_extractor, model);
  //         await tf.browser.toPixels(rgb_img.div(255), canvas);
  //         // img.src = canvas.toDataURL();
  //         // console.log(`${entry.title} is loaded colorize here`);
  //       };
  //       img.crossOrigin = "Anonymous";
  //       img.src = "https:" + entry.image.full;
  //       li_entry.appendChild(img);
  //       image_list.appendChild(li_entry);
  //     });
  //   });
}

tf.tidy(() => {
  main();
});
