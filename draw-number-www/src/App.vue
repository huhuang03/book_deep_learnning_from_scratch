<template>
  <nav>
    <router-link to="/draw">识别</router-link> |
    <router-link to="/create">创建数据</router-link>
  </nav>
  <router-view/>
</template>

<style lang="scss">
html {
  height: 100%;
}
body {
  margin: 0px;
  height: 100%;
}
#app {
  height: 100%;
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  background: #eeeeee;
  color: #2c3e50;
}

.app-root {
  height: 100%;
}

nav {
  padding: 30px;

  a {
    font-weight: bold;
    color: #2c3e50;

    &.router-link-exact-active {
      color: #42b983;
    }
  }
}
</style>
