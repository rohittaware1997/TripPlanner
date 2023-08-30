<template>
  <q-carousel
    v-model="slide"
    animated
    arrows
    navigation
    navigation-icon="radio_button_unchecked"
    control-type="regular"
    control-color="white"
    control-text-color="grey-8"
    height="730px"
  >
    <q-carousel-slide :name="1">
      <!--        <div class="container-grid">-->
      <!--          &lt;!&ndash;    row 1 &ndash;&gt;-->
      <!--          <div @drop="onDrop($event, 1)"-->
      <!--               @dragover.prevent-->
      <!--               @dragenter.prevent class="container-row drop-zone">-->
      <!--            <ResultCard draggable="true" v-for="card in cardObjects(1)" :key="card.id"-->
      <!--                        @dragstart="startDrag($event, card)" :rating-model=card.ratingModel-->
      <!--                        :is-day-activity="card.isDayActivity" :cost-text=card.costText-->
      <!--                        :place-text="card.placeText" :time-text="card.timeText"></ResultCard>-->

      <!--          </div>-->
      <!--        </div>-->

      <div class="container-grid">
        <!--    row 1 -->
        <div  class="container-row drop-zone">
          <div class="container-row row-1" v-for="cards in cat" :key="cards.id">
            <ResultCard draggable="true" v-for="card in cards" :key="card.id"
                        :rating-model=card.rating
                        :is-day-activity="card.timeofday" :cost-text=card.price
                        :place-text="card.name" :time-text="card.category"></ResultCard>
          </div>
        </div>
      </div>
    </q-carousel-slide>
    <q-carousel-slide :name="2">
      <!--        <div class="container-grid">-->
      <!--          &lt;!&ndash;    row 1 &ndash;&gt;-->
      <!--          <div @drop="onDrop($event, 1)"-->
      <!--               @dragover.prevent-->
      <!--               @dragenter.prevent class="container-row drop-zone">-->
      <!--            <ResultCard draggable="true" v-for="card in cardObjects(1)" :key="card.id"-->
      <!--                        @dragstart="startDrag($event, card)" :rating-model=card.ratingModel-->
      <!--                        :is-day-activity="card.isDayActivity" :cost-text=card.costText-->
      <!--                        :place-text="card.placeText" :time-text="card.timeText"></ResultCard>-->

      <!--          </div>-->
      <!--        </div>-->
    </q-carousel-slide>
  </q-carousel>
</template>

<script>
export default {
  name: "demoFile"
};
</script>

<style scoped>
/*const startDrag = (event, card) => {*/
/*                                 //   event.dataTransfer.dropEffect = "move";*/
/*                                 //   event.dataTransfer.effectAllowed = "move";*/
/*                                 //   event.dataTransfer.setData("itemID", card.id);*/
/*                                 // };*/
/*//*/
/*// const onDrop = (event, list) => {*/
/*                                 //   const itemID = event.dataTransfer.getData("itemID");*/
/*                                 //   const item = cardObjectItems.value.find((item) => item.id == itemID);*/
/*                                 //   console.log(list);*/
/*                                 //   item.list = list;*/
/*                                 // };*/
/*//*/
/*// const cardObjects = (list) => {*/
/*                               //   console.log(list)*/
/*                               //   return cardObjectItems.value.filter((item) => item.list == list);*/
/*                               // };*/
</style>
