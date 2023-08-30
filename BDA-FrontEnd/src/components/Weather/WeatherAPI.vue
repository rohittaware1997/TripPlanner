<template>
  <div class="container">
    <div class="card">
      <div class="card-header">
        <i class="fas fa-map-marker-alt"></i>
        <h1 style="padding: 5px"><b> {{ location }}</b></h1>
      </div>
      <div class="card-body">
        <h2><i class="fas fa-thermometer-half"></i> {{ temp }}°F</h2>
        <p class="condition">{{ condition }}</p>
        <div class="details">
          <p><i class="fas fa-thermometer-full"></i> <b>Feels like:</b> {{ feelsLike }}°F</p>
          <p><i class="fas fa-tint"></i> <b>Humidity:</b> {{ humidity }}%</p>
          <p><i class="fas fa-wind"></i> <b>Wind:</b> {{ wind }} mph</p>
        </div>
      </div>
    </div>
  </div>
</template>


<script type="text/javascript" src="./WeatherAPI.js"></script>
<style scoped lang="scss" src="./WeatherAPI.scss"></style>
