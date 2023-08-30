<template>
<!--  first page-->
  <div v-if="pageOne">
    <LandingPage @secondPage="callModel"></LandingPage>
  </div>

<!--  loader/spinner logic -->
  <q-inner-loading label="Please wait we are constructing your plan..."
                   label-class="text-teal"
                   label-style="font-size: 1.1em" :showing="visible">
    <q-spinner-gears size="50px" color="primary" />
  </q-inner-loading>

<!--  second page-->
  <div v-if="pageTwo">
    <ResultPage @backFirstPage="togglePage"></ResultPage>
  </div>
</template>


<style lang="scss" src="./MainLayout.scss"></style>
<script type="text/javascript" src="./MainLayout.js"></script>
