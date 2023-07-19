import { createApp } from 'vue'
import './styles.scss'
import App from './App.vue'
import router from '@/router';
import 'element-plus/dist/index.css'
import ElementPlus from 'element-plus';


createApp(App).use(ElementPlus)
  .use(router)
  .mount('#app')
