<template>
  <section id="timeline">
    <h1>{{ title_1[0] }}</h1>
    <p class="leader">{{ title_1[1] }}</p>
    <div class="demo-card-wrapper">
      <div class="demo-card demo-card--step1">
        <div class="head">
          <div class="number-box">
            <span>01</span>
          </div>
          <h2> {{ heading[0] }} </h2>
        </div>
        <div class="body">
          <p style="text-align: justify">{{ content[0] }}</p>


          <img src="../../assets/Card1_Budget_Based_recommendation.jpg" alt="Graphic">
        </div>
      </div>

      <div class="demo-card demo-card--step2">
        <div class="head">
          <div class="number-box">
            <span>02</span>
          </div>
          <h2> {{ heading[4] }}</h2>
        </div>
        <div class="body">

          <p style="font-weight: bold">Optimizes your itinerary</p>

          <ul class="list">
            <li>Create your own personalized day plan by selecting activities from our top recommendations</li>
            <li>Fully customizable</li>
            <li>Add recommendations to calendar for planning you day</li>
            <li>Download the calendar and save as PDF for future reference</li>

          </ul>
          <img src="../../assets/Card5_Customize_Your_Plan.png" alt="Graphic">
        </div>
      </div>

      <div class="demo-card demo-card--step3" style="background-color: ">
        <div class="head">
          <div class="number-box">
            <span>03</span>
          </div>
          <h2> {{ heading[1] }} </h2>
        </div>
        <div class="body">

          <ul class="lists">
            <li>Play Area to quickly find Distance and Time between places</li>
            <li>Add upto four places which places in better planning</li>
            <li>Each Day has its own instance for getting distance and time</li>
          </ul>
          <img src="../../assets/maps.png" alt="Graphic">
        </div>
      </div>


      <div class="demo-card demo-card--step3">
        <div class="head">
          <div class="number-box">
            <span>04</span>
          </div>
          <h2> {{ heading[2] }}</h2>
        </div>
        <div class="body">
          <p> {{ content[1] }}</p>
          <img src="../../assets/Card3_Day_Night.jpg" alt="Graphic">
        </div>
      </div>

      <div class="demo-card demo-card--step4">
        <div class="head">
          <div class="number-box">
            <span>05</span>
          </div>
          <h2> {{ heading[3] }}</h2>
        </div>
        <div class="body">
          <p>{{ content[2] }}
          </p>
          <img src="../../assets/Card4_Top_Recommendation.jpg" alt="Graphic">
        </div>
      </div>



    </div>
  </section>
</template>

<script type="text/javascript" src="./FeatureTimeline.js"></script>
<style scoped lang="scss" src="./FeatureTimeline.scss"></style>
