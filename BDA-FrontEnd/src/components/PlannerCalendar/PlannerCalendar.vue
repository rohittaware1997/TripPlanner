<template>
  <div :style="{height: height + 'px'}">
    <vue-cal :time-step="60" :events="events"
             :selected-date="sDate"
             :editable-events="{ title: true, drag: true, resize: true, delete: true, create: false }"
             cell-contextmenu

             class=" vuecal vuecal--green-theme ex--disable-views vuecal--full-height-delete"
             @event-delete="delEvent('event-delete', $event)"
             :time="true" active-view="day" :disable-views="['years', 'year', 'month','week']">
    </vue-cal>
  </div>
  <div>
  </div>

</template>
<script type="text/javascript" src="./PlannerCalendar.js"></script>
<style scoped lang="scss" src="./PlannerCalendar.scss"></style>
