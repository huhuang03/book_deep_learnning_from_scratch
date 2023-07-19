<template>
  <div class="draw-page-root">
    <div class="recognize-container">
      <div style="flex: 1"></div>
      <canvas ref="c"/>
      <div class="result" style="flex: 1; margin-left: 30px">
        {{recognizedResult}}
      </div>
    </div>
    <div class="op-container">
      <button @click="handleClickClear">清空</button>
    </div>
    <div style="margin-top: 20px">
      <div v-for="item in models" :key="item.name" style="display: flex; justify-content: center; margin-top: 10px">
        <div style="min-width: 300px; text-align: left; display: flex; align-items: center">
          {{item.name}}
          <el-tooltip :content="item.desc" v-if="item.desc">
            <el-icon style="margin-left: 5px"><Warning /></el-icon>
          </el-tooltip>
        </div>
        <button class="bt-recognize" @click="handleRecognizeClick(item)">{{recognizedText}}</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {onMounted, ref} from 'vue';
import {Draw} from '@/DrawCanvas';
import axios from 'axios';
import {ElMessage} from 'element-plus';
import {Warning} from '@element-plus/icons-vue';

const canvas = ref<Draw>()
const c = ref<HTMLCanvasElement>()
const cursor = ref()

const recognizedText = ref('识别')
const recognizedResult = ref('')

let location = window.location;
const apiHref = `${location.protocol}//${location.hostname}:8001/api`
const models = ref([])

function handleClickClear() {
  canvas.value!.clear()
}

function handleRecognizeClick(item: any) {
  recognizedText.value = '识别中'
  const data = c.value?.toDataURL('image/jpeg')

  axios.post(`${apiHref}/recognize`, {
    img: data,
    model: item.name
  }).then(res => {
    const data = res.data.data
    const rst = data.possible.map((item: number, index: number) => ({num: index, possible: item}))
    recognizedResult.value = `结果: ${data.rst}\n` + rst.map((item: any) => `${item.num}: possible: ${item.possible}`).join('\n')
    console.log('rst: ', rst)
  }).catch(err => {
    recognizedResult.value = '识别失败'
  }).finally(() => {
    recognizedText.value = '识别'
  })

}

onMounted(() => {
  canvas.value = new Draw(c.value!, cursor.value)

  axios.get(`${apiHref}/list`).then(res => {
    if (res.data.code !== 0) {
      ElMessage('获取模型失败')
      return
    }
    models.value = res.data.data
  }).catch(err => {
    ElMessage('获取模型失败')
  })
})
</script>

<style lang="scss">
.draw-page-root {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 100%;
  background: #eeeeee;

  canvas {
    width: 200px;
    height: 200px;
    background: black;
  }
  .cursor {
    position: fixed;
    top: 0;
    left: 0;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: 3px solid rgb(30, 30, 30);
    pointer-events: none;
    user-select: none;
    mix-blend-mode: difference;
    opacity: 0;
    transition: opacity 1s;
  }

  .bt-recognize {
    margin-left: 10px;
  }

  .op-container {
    margin-top: 10px;
  }

  .result {
    text-align: left;
    margin-top: 10px;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: inherit;
  }

  .recognize-container {
    width: 100%;
    display: flex;
    flex-direction: row;
  }
}
</style>
