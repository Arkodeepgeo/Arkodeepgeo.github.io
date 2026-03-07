# Zn–Pb Deposit Classification

Categorization of Zn–Pb deposits based on sphalerite geochemistry using machine learning.

## Project Description

This project uses sphalerite trace element geochemistry to classify Zn–Pb deposit types using machine learning models such as Random Forest, XGBoost, SVM, AdaBoost and Stacking.

## Code Repository

[View GitHub Code](https://github.com/Arkodeepgeo)

## Web Application

[Open Web App](https://your-shiny-webapp-link)

## Workflow

![Workflow](workflow.png)

## Figures from the Study
<div class="slider-container">

<img class="slides" src="Figures/Figure_1.jpg">
<img class="slides" src="Figures/Figure_2.jpg">
<img class="slides" src="Figures/Figure_3.jpg">
<img class="slides" src="Figures/Figure_4.jpg">
<img class="slides" src="Figures/Figure_5.jpg">
<img class="slides" src="Figures/Figure_6.jpg">
<img class="slides" src="Figures/Figure_7.jpg">
<img class="slides" src="Figures/Figure_8.jpg">
<img class="slides" src="Figures/Figure_9.jpg">
<img class="slides" src="Figures/Figure_10.jpg">

</div>

<style>
.slider-container{
  max-width:900px;
  margin:auto;
}

.slides{
  width:100%;
  display:none;
}
</style>

<script>
let slideIndex = 0;
showSlides();

function showSlides(){

let slides = document.getElementsByClassName("slides");

for(let i=0;i<slides.length;i++){
slides[i].style.display="none";
}

slideIndex++;

if(slideIndex > slides.length){
slideIndex = 1;
}

slides[slideIndex-1].style.display="block";

setTimeout(showSlides,2500);
}
</script>
