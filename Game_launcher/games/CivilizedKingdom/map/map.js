let map_size = 30; // character.js와 동일하게 30으로 고정
let tile_size = 48;

function drawMap() {
  const canvas = document.getElementById('map_layer');
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#222';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // 맵 격자(테두리)만 예시로 그림
  for (let x = 0; x < map_size; x++) {
    for (let y = 0; y < map_size; y++) {
      let px = (x - y) * tile_size / 2 + canvas.width / 2;
      let py = (x + y) * tile_size * 0.3;
      ctx.strokeStyle = '#555';
      ctx.strokeRect(px - tile_size/2, py - tile_size/2, tile_size, tile_size);
    }
  }
} 