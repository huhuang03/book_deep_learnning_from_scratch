<template>
  <div class="create-page-root" @keydown="handleKeyDown">
    <el-radio-group
        v-model="num">
      <el-radio v-for="item of numbers"
                style="margin-left: 20px"
                size="large"
                :label="item"
                :key="item.toString()">{{item}}</el-radio>
    </el-radio-group>

    <div style="margin-top: 20px; font-size: 20px">当前数字：<span style="color: red">{{num}}</span></div>

    <canvas style="margin-top: 20px" ref="c"
            @contextmenu="handleRightClick" />
    <div class="cursor" ref="cursor"></div>
    <div class="op-container">
      <button @click="handleClickClear">清空</button>
      <button class="bt-recognize" @click="handleRecognizeClick">{{ commitText }}</button>
    </div>
    <div class="result">
      {{ commitResult }}
    </div>
  </div>
</template>

<script setup lang="ts">
import {onMounted, ref} from 'vue';
import {Draw} from '@/DrawCanvas';
import axios from 'axios';
import {ElMessage} from 'element-plus';
const numbers = [...Array(10).keys()]
const num = ref(0)

const canvas = ref<Draw>()
const c = ref<HTMLCanvasElement>()
const cursor = ref()

const commitText = ref('提交')
const commitResult = ref('')
const hasChanged = ref(true)

let location = window.location;
const apiHref = `${location.protocol}//${location.hostname}:8001/api`
function handleClickClear() {
  canvas.value!.clear()
  hasChanged.value = true
}

function handleKeyDown(event: any) {
  console.log('event: ', event)
}

function handleRightClick(event: any) {
  event.preventDefault()
  handleRecognizeClick()
}

function handleRecognizeClick() {
  if (canvas.value!.isClear) {
    ElMessage('没有内容，无法提交')
    return
  }

  if (!hasChanged.value) {
    commitResult.value = '已经提交过了'
    return
  }

  hasChanged.value = false
  commitText.value = '提交中'
  const data = c.value?.toDataURL('image/jpeg')

  axios.post(`${apiHref}/create`, {
    img: data,
    num: num.value
  }).then(res => {
    const code = res.data.code
    if (code === 0) {
      commitResult.value = '提交成功'
      canvas.value!.clear()
      hasChanged.value = true
    } else {
      commitResult.value = '提交失败'
    }
  }).catch(err => {
    commitResult.value = '提交失败'
  }).finally(() => {
    commitText.value = '提交'
  })

}

onMounted(() => {
  canvas.value = new Draw(c.value!, () => {
    hasChanged.value = true
  })
})
</script>

<style lang="scss">
.create-page-root {
  height: 100%;
  width: 100%;
  background: #eeeeee;

  canvas {
    width: 200px;
    height: 200px;
    background: black;
  }

  .bt-recognize {
    margin-left: 10px;
  }

  .op-container {
    margin-top: 10px;
  }

  .result {
    margin-top: 10px;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: inherit;
  }
}
</style>
