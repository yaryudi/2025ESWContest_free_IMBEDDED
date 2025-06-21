let units = [
  { id: 1, x: 6, y: 12, type: 'warrior', tribe: 'Xin-xi', moves_left: 10000, label: '주인공' },
  { id: 2, x: 8, y: 12, type: 'warrior', tribe: 'Imperius', moves_left: 10000, label: '동료1' },
  { id: 3, x: 6, y: 14, type: 'warrior', tribe: 'Bardur', moves_left: 10000, label: '동료2' },
  { id: 4, x: 8, y: 14, type: 'warrior', tribe: 'Oumaji', moves_left: 10000, label: '동료3' }
];
let tile_size = 48;

function getMapSize() {
  // 항상 window.map_size를 우선 사용, 없으면 30
  return (typeof window.map_size === 'number' && !isNaN(window.map_size)) ? window.map_size : 30;
}

function drawCharacters() {
  const canvas = document.getElementById('character_layer');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  let map_size = getMapSize();

  for (let unit of units) {
    // map.js와 완전히 동일하게 변환 (보정 없이)
    let px = (unit.x - unit.y) * tile_size / 2 + canvas.width / 2;
    let py = (unit.x + unit.y) * tile_size * 0.3;

    ctx.fillStyle = '#ff0';
    ctx.fillRect(px - 20, py - 20, 40, 40);

    ctx.fillStyle = '#fff';
    ctx.font = '20px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(unit.id, px, py);

    ctx.font = '16px Arial';
    ctx.fillStyle = '#0ff';
    ctx.fillText(unit.label, px, py + 32);
  }
}

function moveUnit(unitId, dx, dy) {
  let unit = units.find(u => u.id === unitId);
  if (!unit || unit.moves_left <= 0) return;
  let newX = unit.x + dx;
  let newY = unit.y + dy;
  let map_size = getMapSize();
  if (newX < 0 || newX >= map_size || newY < 0 || newY >= map_size) return;
  if (units.some(u => u.id !== unitId && u.x === newX && u.y === newY)) return;
  unit.x = newX;
  unit.y = newY;
  unit.moves_left--;
  drawCharacters();
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'ArrowUp') moveUnit(1, 0, -1);
  if (e.key === 'ArrowDown') moveUnit(1, 0, 1);
  if (e.key === 'ArrowLeft') moveUnit(1, -1, 0);
  if (e.key === 'ArrowRight') moveUnit(1, 1, 0);
}); 