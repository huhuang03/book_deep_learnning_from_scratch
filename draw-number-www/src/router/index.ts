import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'
import DrawPage from '@/module/draw/DrawPage.vue';
import CreatePage from '@/module/create/CreatePage.vue';

const routes: Array<RouteRecordRaw> = [
  {
    path: '/draw',
    name: 'draw',
    component: DrawPage
  },
  {
    path: '/create',
    name: 'create',
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () => import(/* webpackChunkName: "about" */ '../module/create/CreatePage.vue')
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
