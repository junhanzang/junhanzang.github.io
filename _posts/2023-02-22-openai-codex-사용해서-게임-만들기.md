---
title: "OpenAI Codex 사용해서 게임 만들기"
date: 2023-02-22 14:58:14
categories:
  - 프로젝트
tags:
  - OpenAI Codex
---

<https://openai.com/blog/openai-codex/>

[OpenAI Codex](https://openai.com/blog/openai-codex/)

```
/* /* Create the canvas element. */ */
var canvas = document.createElement('canvas');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
document.body.appendChild(canvas);
/* Get the canvas context. */
var ctx = canvas.getContext('2d');
/* Set the canvas size. */
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
/* Add the canvas to the body element. */
document.body.appendChild(canvas);
/* Create a player object in the lower center */
var player = {
  x: canvas.width / 2,
  y: canvas.height - 50,
  width: 50,
  height: 50,
  color: '#00A',
  draw: function() {
    ctx.fillStyle = this.color;
    ctx.fillRect(this.x - this.width / 2, this.y - this.height / 2, this.width, this.height);
  }
};
/* A variable that stores the state of moving left. */
var moveLeft = false;
/* A variable that stores the state of moving right. */
var moveRight = false;
/* An array of flower objects */
var flowers = [];
/* A variable to store the score. */
var score = 0;
/* function creates a new flower. */
function createFlower() {
  var flower = {
    x: Math.random() * canvas.width,
    y: 0,
    width: 50,
    height: 50,
    color: '#A00',
    draw: function() {
      ctx.fillStyle = this.color;
      ctx.fillRect(this.x - this.width / 2, this.y - this.height / 2, this.width, this.height);
    }
  };
  flowers.push(flower);
}
/* A function that draws the player. */
function drawPlayer() {
  player.draw();
}
/* A function that draws flowers. */
function drawFlowers() {
  flowers.forEach(function(flower) {
    flower.draw();
  });
}
/* A function that moves the player. */
function movePlayer() {
  if (moveLeft) {
    player.x -= 5;
  }
  if (moveRight) {
    player.x += 5;
  }
}
/* This function moves the flower. */
function moveFlowers() {
  flowers.forEach(function(flower) {
    flower.y += 5;
  });
}
/* A function that checks for collisions. */
function checkCollisions() {
  flowers.forEach(function(flower) {
    if (
      player.x < flower.x + flower.width / 2 &&
      player.x + player.width / 2 > flower.x &&
      player.y < flower.y + flower.height / 2 &&
      player.y + player.height / 2 > flower.y
    ) {
      score++;
      flowers.splice(flowers.indexOf(flower), 1);
    }
  });
}
/* A function that draws scores. */
function drawScore() {
  ctx.font = '24px Helvetica';
  ctx.fillStyle = '#00A';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('Score: ' + score, 32, 32);
}
/* This function checks for game over. */
function checkGameOver() {
  flowers.forEach(function(flower) {
    if (flower.y > canvas.height) {
      flowers.splice(flowers.indexOf(flower), 1);
      score--;
    }
  });
}
/* This function runs the game loop */
function gameLoop() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawPlayer();
  drawFlowers();
  movePlayer();
  moveFlowers();
  checkCollisions();
  drawScore();
  checkGameOver();
  requestAnimationFrame(gameLoop);
}
/* Register keyboard event listeners. */
document.addEventListener('keydown', function(event) {
  if (event.keyCode === 37) {
    moveLeft = true;
  } else if (event.keyCode === 39) {
    moveRight = true;
  }
});
document.addEventListener('keyup', function(event) {
  if (event.keyCode === 37) {
    moveLeft = false;
  } else if (event.keyCode === 39) {
    moveRight = false;
  }
});
/* Start the game loop. */
gameLoop();
/* Create a new flower every second. */
setInterval(createFlower, 1000);
```

주석을 입력하면 바로 아래에 코드를 만들어준다.

[avoid cherry blossoms.html

0.01MB](./file/avoid cherry blossoms.html)

html로 변환도 해준다.
