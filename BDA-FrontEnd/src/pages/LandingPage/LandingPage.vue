<template>
  <!--  <GoogleMap></GoogleMap>-->
  <!--  <Weather></Weather>-->
  <div class="parallax" id="home">
    <ParallaxScroll></ParallaxScroll>
    <div class="parallax__cover">
      <header class="title">
        Welcome to Trip Planner
        <q-btn class="btn-title" @click="jumpToElement('search')" outline rounded color="white" label="Let's Begin" />
      </header>
      <div class="header">
        <h2 class="logo">Trip Planner</h2>
        <nav class="navigation">
          <a @click="jumpToElement('home')" class="active">Home</a>
          <a @click="jumpToElement('search')">Search</a>
          <a @click="jumpToElement('feature')">Features</a>
        </nav>
      </div>
      <div class="q-pa-md stepper-progress" id="search">
        <q-stepper
          class="stepper-progress-1"
          v-model="step"
          ref="stepper"
          color="primary"
          alternative-labels
          animated>
          <q-step
            :name="1"
            title="Basic Input"
            icon="edit"
            :done="step > 1">
            <SearchBar style="" @visibleChanged="handleValueChanged" v-if="nextParam"></SearchBar>
          </q-step>
          <q-step
            :name="2"
            title="Select five category"
            icon="category"
            :done="step > 2">
            <transition class="fade">
              <TileSelection style="padding:20px; " v-if="nextParam"
                             :attraction_types="attraction_types" :slider-status="true"></TileSelection>
            </transition>
          </q-step>
<!--          <q-step-->
<!--            :name="3"-->
<!--            title="Select five hotel preferences"-->
<!--            icon="hotel"-->
<!--            :done="step > 3">-->
<!--            <transition class="fade">-->
<!--              <TileSelection class="" style="padding:20px; " v-if="nextParam"-->
<!--                             :attraction_types="hotel_types" :slider-status="false"></TileSelection>-->
<!--            </transition>-->
<!--          </q-step>-->
          <q-step
            :name="3"
            title="Done"
            icon="verified"
            :done="step > 3">
            <transition-group class="fade" >
              <div style="padding: 100px">
                <p style="text-align: center; font-size: 2em;">You are all set. Click the button to get your recommendations</p>
                <q-btn  style="font-size: 1.2em; margin-left: 700px" @click="triggerFunction()" color="primary" label="Click Here" />
              </div>

            </transition-group>
          </q-step>
          <template v-slot:navigation>
            <q-stepper-navigation>
              <q-btn style="margin-left: 30px; margin-bottom: 5px; font-size: 1.12em;" @click="$refs.stepper.next();"
                     color="primary" v-if="step!==3" :label="step === 2 ? 'Finish' : 'Continue'" />
              <q-btn style="font-size: 1.12em;" v-if="step > 1 && step !== 3" flat color="primary" @click="$refs.stepper.previous()"
                     label="Back"
                     class="q-ml-sm" />
            </q-stepper-navigation>
          </template>
        </q-stepper>
      </div>
      <div id="feature">
        <FeatureTimeline style="margin-top: 10px"></FeatureTimeline>
      </div>
    </div>
  </div>
</template>

<script type="text/javascript" src="./LandingPage.js"></script>
<style scoped lang="scss" src="./LandingPage.scss"></style>
