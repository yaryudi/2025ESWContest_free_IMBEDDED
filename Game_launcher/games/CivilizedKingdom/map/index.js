let page = 'main';
let map_size;
let map;
let tile_size = 77;  // 기본 타일 크기 (1.6배)
let width = 1280;    // 기본 캔버스 너비 (1.6배)
let height = 960;    // 기본 캔버스 높이 (1.6배)
let canvas = null;   // 캔버스 컨텍스트

const tribes_list = ['Xin-xi', 'Imperius', 'Bardur', 'Oumaji', 'Kickoo', 'Hoodrick', 'Luxidoor', 'Vengir', 'Zebasi',
    'Ai-mo', 'Quetzali', 'Yadakk', 'Aquarion', 'Elyrion', 'Polaris'];
const terrain = ['forest', 'fruit', 'game', 'ground', 'mountain'];
const general_terrain = ['crop', 'fish', 'metal', 'ocean', 'ruin', 'village', 'water', 'whale'];
const $$$$$ = 2;
const $$$$ = 1.5;
const ___ = 1;
const $$ = 0.5;
const $ = 0.1;
const BORDER_EXPANSION = 1/3;
const terrain_probs = {'water': {'Xin-xi': 0, 'Imperius': 0, 'Bardur': 0, 'Oumaji': 0, 'Kickoo': 0.4, 'Hoodrick': 0, 'Luxidoor': 0,
                        'Vengir': 0, 'Zebasi': 0, 'Ai-mo': 0, 'Quetzali': 0, 'Yadakk': 0, 'Aquarion': 0.3, 'Elyrion': 0},
                    'forest': {'Xin-xi': ___, 'Imperius': ___, 'Bardur': ___, 'Oumaji': $, 'Kickoo': ___, 'Hoodrick': $$$$, 'Luxidoor': ___,
                        'Vengir': ___, 'Zebasi': $$, 'Ai-mo': ___, 'Quetzali': ___, 'Yadakk': $$, 'Aquarion': $$, 'Elyrion': ___},
                    'mountain': {'Xin-xi': $$$$, 'Imperius': ___, 'Bardur': ___, 'Oumaji': ___, 'Kickoo': $$, 'Hoodrick': $$, 'Luxidoor': ___,
                        'Vengir': ___, 'Zebasi': $$, 'Ai-mo': $$$$, 'Quetzali': ___, 'Yadakk': $$, 'Aquarion': ___, 'Elyrion': $$},
                    'metal': {'Xin-xi': ___, 'Imperius': ___, 'Bardur': ___, 'Oumaji': ___, 'Kickoo': ___, 'Hoodrick': ___, 'Luxidoor': ___,
                        'Vengir': $$$$$, 'Zebasi': ___, 'Ai-mo': ___, 'Quetzali': $, 'Yadakk': ___, 'Aquarion': ___, 'Elyrion': ___},
                    'fruit': {'Xin-xi': ___, 'Imperius': ___, 'Bardur': $$$$, 'Oumaji': ___, 'Kickoo': ___, 'Hoodrick': ___, 'Luxidoor': $$$$$,
                        'Vengir': $, 'Zebasi': $$, 'Ai-mo': ___, 'Quetzali': $$$$$, 'Yadakk': $$$$, 'Aquarion': ___, 'Elyrion': ___},
                    'crop': {'Xin-xi': ___, 'Imperius': ___, 'Bardur': $, 'Oumaji': ___, 'Kickoo': ___, 'Hoodrick': ___, 'Luxidoor': ___,
                        'Vengir': ___, 'Zebasi': ___, 'Ai-mo': $, 'Quetzali': $, 'Yadakk': ___, 'Aquarion': ___, 'Elyrion': $$$$},
                    'game': {'Xin-xi': ___, 'Imperius': ___, 'Bardur': $$$$$, 'Oumaji': ___, 'Kickoo': ___, 'Hoodrick': ___, 'Luxidoor': $$,
                        'Vengir': $, 'Zebasi': ___, 'Ai-mo': ___, 'Quetzali': ___, 'Yadakk': ___, 'Aquarion': ___, 'Elyrion': ___},
                    'fish': {'Xin-xi': ___, 'Imperius': ___, 'Bardur': ___, 'Oumaji': ___, 'Kickoo': $$$$, 'Hoodrick': ___, 'Luxidoor': ___,
                        'Vengir': $, 'Zebasi': ___, 'Ai-mo': ___, 'Quetzali': ___, 'Yadakk': ___, 'Aquarion': ___, 'Elyrion': ___},
                    'whale': {'Xin-xi': ___, 'Imperius': ___, 'Bardur': ___, 'Oumaji': ___, 'Kickoo': ___, 'Hoodrick': ___, 'Luxidoor': ___,
                        'Vengir': ___, 'Zebasi': ___, 'Ai-mo': ___, 'Quetzali': ___, 'Yadakk': ___, 'Aquarion': ___, 'Elyrion': ___}};
const general_probs = {'mountain': 0.15, 'forest': 0.4, 'fruit': 0.5, 'crop': 0.5, 'fish': 0.5, 'game': 0.5, 'whale': 0.4, 'metal': 0.5};

let assets = [];
let get_assets = new Promise(resolve => {
    for (let tribe of tribes_list) {
        assets[tribe] = [];
    }
    for (let g_t of general_terrain) {
        assets[g_t] = get_image("assets/" + g_t + ".png");
    }
    for (let tribe of tribes_list) {
        for (let terr of terrain) {
            assets[tribe][terr] = get_image("assets/" + tribe + "/" + tribe + " " + terr + ".png");
        }
        assets[tribe]['capital'] = get_image("assets/" + tribe + "/" + tribe + " head.png");
    }
    resolve();
});

window.addEventListener('resize', resizeCanvas);

function initializeCanvas() {
    const graphic_display = document.getElementById("graphic_display");
    if (!graphic_display) return false;
    
    canvas = graphic_display.getContext("2d");
    if (!canvas) return false;
    
    width = graphic_display.width;
    height = graphic_display.height;
    return true;
}

function onload() {
    get_assets.then(() => {
        if (initializeCanvas()) {
            resizeCanvas(); // 캔버스 크기 동기화
            generate();
        }
    });
}

function resizeCanvas() {
    const canvas = document.getElementById("graphic_display");
    if (!canvas) return;
    
    const parent = canvas.parentElement;
    if (!parent) return;

    width = parent.clientWidth;
    height = parent.clientHeight;
    canvas.width = width;
    canvas.height = height;

    if (typeof map !== 'undefined') {
        display_map(map);
    }
}

function switch_page(new_page) {
    document.getElementById("main").style.display='none';
    document.getElementById("faq").style.display='none';
    page = new_page;
    document.getElementById(new_page).style.display='block';
}

function get_image(src) {
    let image = new Image();
    image.src = src;
    return image;
}

function generate() {
    // 입력란이 없으면 기본값 사용
    let map_size_elem = document.getElementById("map_size");
    map_size = map_size_elem ? parseInt(map_size_elem.value) : 30;

    let initial_land_elem = document.getElementById("initial_land");
    let initial_land = initial_land_elem ? parseFloat(initial_land_elem.value) : 0.5;

    let smoothing_elem = document.getElementById("smoothing");
    let smoothing = smoothing_elem ? parseInt(smoothing_elem.value) : 3;

    let relief_elem = document.getElementById("relief");
    let relief = relief_elem ? parseInt(relief_elem.value) : 4;

    let tribes_elem = document.getElementById("tribes");
    let tribes = tribes_elem ? tribes_elem.value : "Xin-xi Imperius Bardur Oumaji";
    tribes = tribes.split(" ");

    let fill_elem = document.getElementById("fill");
    let fill = fill_elem ? fill_elem.value : "";

    let no_biomes_elem = document.getElementById("no_biomes_check");
    let no_biomes_check = no_biomes_elem ? no_biomes_elem.checked : false;

    let no_resources_elem = document.getElementById("no_resources_check");
    let no_resources_check = no_resources_elem ? no_resources_elem.checked : false;

    document.getElementById("warning") && (document.getElementById("warning").style.display='none');

    // let the show begin
    console.time('Initial map');
    let land_coefficient = (0.5 + relief) / 9;
    map = new Array(map_size**2);

    // add initial ocean tiles
    for (let i = 0; i < map_size**2; i++) {
        map[i] = {type: 'ocean', above: null, road: false, tribe: fill ? fill : 'Xin-xi'}; // tribes don't matter so far
    }

    // randomly replace half of the tiles with ground
    let i = 0;
    while (i < map_size**2 * initial_land) {
        let cell = random_int(0, map_size**2);
        if (map[cell]['type'] === 'ocean') {
            i++;
            map[cell]['type'] = 'ground';
        }
    }
    console.timeEnd('Initial map');

    // turning random water/ground grid into something smooth
    console.time('Smoothing');
    for (let i = 0; i < smoothing; i++) {
        for (let cell = 0; cell < map_size**2; cell++) {
            let water_count = 0;
            let tile_count = 0;
            let neighbours = round(cell, 1);
            for (let i = 0; i < neighbours.length; i++) {
                if (map[neighbours[i]]['type'] === 'ocean') {
                    water_count++;
                }
                tile_count++;
            }
            if (water_count / tile_count <= land_coefficient)
                map[cell]['road'] = true; // mark as a road if it has to be ground (most of the neighbours are ground)
        }
        // turn marked tiles into ground & everything else into water
        for (let cell = 0; cell < map_size**2; cell++) {
            if (map[cell]['road'] === true) {
                map[cell]['road'] = false;
                map[cell]['type'] = 'ground';
            } else {
                map[cell]['type'] = 'ocean';
            }
        }
    }
    console.timeEnd('Smoothing');

    // capital distribution
    let capital_cells = [];
    if (!fill) {
        console.time('Capital distribution');
        let capital_map = {};
        // make a map of potential (ground) tiles associated with numbers (0 by default)
        for (let tribe of tribes) {
            for (let row = 2; row < map_size - 2; row++) {
                for (let column = 2; column < map_size - 2; column++) {
                    if (map[row * map_size + column]['type'] === 'ground') {
                        capital_map[row * map_size + column] = 0;
                    }
                }
            }
        }
        for (let tribe of tribes) {
            let max = 0;
            // this number is a sum of distances between the tile and all capitals
            for (let cell in capital_map) {
                capital_map[cell] = map_size;
                for (let capital_cell of capital_cells) {
                    capital_map[cell] = Math.min(capital_map[cell], distance(cell, capital_cell, map_size));
                }
                max = Math.max(max, capital_map[cell]);
            }
            let len = 0;
            for (let cell in capital_map) {
                if (capital_map[cell] === max) {
                    len++;
                }
            }
            // we want to find a tile with a maximum sum
            let rand_cell = random_int(0, len);
            for (let cell of Object.entries(capital_map)) {
                if (cell[1] === max) {
                    if (rand_cell === 0) {
                        capital_cells.push(parseInt(cell[0]));
                    }
                    rand_cell--;
                }
            }
        }
        for (let i = 0; i < capital_cells.length; i++) {
            map[(capital_cells[i] / map_size | 0) * map_size + (capital_cells[i] % map_size)]['above'] = 'capital';
            map[(capital_cells[i] / map_size | 0) * map_size + (capital_cells[i] % map_size)]['tribe'] = tribes[i];
        }
        console.timeEnd('Capital distribution');
    }

    // terrain distribution
    if (!fill) {
        console.time('Terrain distribution');
        let done_tiles = [];
        let active_tiles = []; // done tiles that generate terrain around them
        for (let i = 0; i < capital_cells.length; i++) {
            done_tiles[i] = capital_cells[i];
            active_tiles[i] = [capital_cells[i]];
        }
        // we'll start from capital tiles and evenly expand until the whole map is covered
        while (done_tiles.length !== map_size**2) {
            for (let i = 0; i < tribes.length; i++) {
                if (active_tiles[i].length && tribes[i] !== 'Polaris') {
                    let rand_number = random_int(0, active_tiles[i].length);
                    let rand_cell = active_tiles[i][rand_number];
                    let neighbours = circle(rand_cell, 1);
                    let valid_neighbours = neighbours.filter(value => done_tiles.indexOf(value) === -1 && map[value]['type'] !== 'water');
                    if (!valid_neighbours.length) {
                        valid_neighbours = neighbours.filter(value => done_tiles.indexOf(value) === -1);
                    } // if there are no land tiles around, accept water tiles
                    if (valid_neighbours.length) {
                        let new_rand_number = random_int(0, valid_neighbours.length);
                        let new_rand_cell = valid_neighbours[new_rand_number];
                        map[new_rand_cell]['tribe'] = tribes[i];
                        active_tiles[i].push(new_rand_cell);
                        done_tiles.push(new_rand_cell);
                    } else {
                        active_tiles[i].splice(rand_number, 1); // deactivate tiles surrounded with done tiles
                    }
                }
            }
        }
        console.timeEnd('Terrain distribution');
    }

    // generate forest, mountains, and extra water according to terrain underneath
    if (!no_biomes_check) {
        console.time('Biome generation');
        for (let cell = 0; cell < map_size**2; cell++) {
            if (map[cell]['type'] === 'ground' && map[cell]['above'] === null) {
                let rand = Math.random(); // 0 (---forest---)--nothing--(-mountain-) 1
                if (rand < general_probs['forest'] * terrain_probs['forest'][map[cell]['tribe']]) {
                    map[cell]['type'] = 'forest';
                } else if (rand > 1 - general_probs['mountain'] * terrain_probs['mountain'][map[cell]['tribe']]) {
                    map[cell]['type'] = 'mountain';
                }
                rand = Math.random(); // 0 (---water---)--------nothing-------- 1
                if (rand < terrain_probs['water'][map[cell]['tribe']]) {
                    map[cell]['type'] = 'ocean';
                }
            }
        }
        console.timeEnd('Biome generation');
    }

    // -1 - water far away
    // 0 - far away
    // 1 - border expansion
    // 2 - initial territory
    // 3 - village
    let village_map = [];
    if (!fill) {
        console.time('Initial village map');
        for (let cell = 0; cell < map_size**2; cell++) {
            let row = cell / map_size | 0;
            let column = cell % map_size;
            if (map[cell]['type'] === 'ocean' || map[cell]['type'] === 'mountain') {
                village_map[cell] = -1;
            } else if (row === 0 || row === map_size - 1 || column === 0 || column === map_size - 1) {
                village_map[cell] = -1; // villages don't spawn next to the map border
            } else {
                village_map[cell] = 0;
            }
        }
        console.timeEnd('Initial village map');
    }
    // we'll place villages until there are none of 'far away' tiles

    // replace some ocean with shallow water
    console.time('Shallow water');
    let land_like_terrain = ['ground', 'forest', 'mountain'];
    for (let cell = 0; cell < map_size**2; cell++) {
        if (map[cell]['type'] === 'ocean') {
            for (let neighbour of plus_sign(cell)) {
                if (land_like_terrain.indexOf(map[neighbour]['type']) !== -1) {
                    map[cell]['type'] = 'water';
                    break;
                }
            }
        }
    }
    console.timeEnd('Shallow water');

    // mark tiles next to capitals according to the notation
    let village_count = 0;
    if (!fill) {
        console.time('Village map generation');
        for (let capital of capital_cells) {
            village_map[capital] = 3;
            for (let cell of circle(capital, 1)) {
                village_map[cell] = Math.max(village_map[cell], 2);
            }
            for (let cell of circle(capital, 2)) {
                village_map[cell] = Math.max(village_map[cell], 1);
            }
        }

        // generate villages & mark tiles next to them
        while (village_map.indexOf(0) !== -1) {
            let new_village = rand_array_element(village_map.map((cell, index) => cell === 0 ? index : null).filter(cell => cell !== null));
            village_map[new_village] = 3;
            for (let cell of circle(new_village, 1)) {
                village_map[cell] = Math.max(village_map[cell], 2);
            }
            for (let cell of circle(new_village, 2)) {
                village_map[cell] = Math.max(village_map[cell], 1);
            }
            village_count++;
        }
        console.timeEnd('Village map generation');
    }

    function proc(cell, probability) {
        return (village_map[cell] === 2 && Math.random() < probability) || (village_map[cell] === 1 && Math.random() < probability * BORDER_EXPANSION)
    }

    // generate resources
    if (!no_resources_check && !no_biomes_check && !fill) {
        console.time('Resource generation');
        for (let cell = 0; cell < map_size**2; cell++) {
            switch (map[cell]['type']) {
                case 'ground':
                    let fruit = general_probs['fruit'] * terrain_probs['fruit'][map[cell]['tribe']];
                    let crop = general_probs['crop'] * terrain_probs['crop'][map[cell]['tribe']];
                    if (map[cell]['above'] !== 'capital') {
                        if (village_map[cell] === 3) {
                            map[cell]['above'] = 'village';
                        } else if (proc(cell, fruit * (1 - crop / 2))) {
                            map[cell]['above'] = 'fruit';
                        } else if (proc(cell, crop * (1 - fruit / 2))) {
                            map[cell]['above'] = 'crop';
                        }
                    }
                    break;
                case 'forest':
                    if (map[cell]['above'] !== 'capital') {
                        if (village_map[cell] === 3) {
                            map[cell]['type'] = 'ground';
                            map[cell]['above'] = 'village';
                        } else if (proc(cell, general_probs['game'] * terrain_probs['game'][map[cell]['tribe']])) {
                            map[cell]['above'] = 'game';
                        }
                    }
                    break;
                case 'water':
                    if (proc(cell, general_probs['fish'] * terrain_probs['fish'][map[cell]['tribe']])) {
                        map[cell]['above'] = 'fish';
                    }
                    break;
                case 'ocean':
                    if (proc(cell, general_probs['whale'] * terrain_probs['whale'][map[cell]['tribe']])) {
                        map[cell]['above'] = 'whale';
                    }
                    break;
                case 'mountain':
                    if (proc(cell, general_probs['metal'] * terrain_probs['metal'][map[cell]['tribe']])) {
                        map[cell]['above'] = 'metal';
                    }
                    break;
            }
        }
        console.timeEnd('Resource generation');
    }

    // ruins generation
    let ruins_number;
    if (!fill) {
        console.time('Ruin generation');
        ruins_number = Math.round(map_size**2/40);
        let water_ruins_number = Math.round(ruins_number/3);
        let ruins_count = 0;
        let water_ruins_count = 0;
        while (ruins_count < ruins_number) {
            let ruin = rand_array_element(village_map.map((cell, index) => cell === 0 || cell === 1 || cell === -1 ? index : null).filter(cell => cell !== null));
            let terrain = map[ruin].type;
            if (terrain !== 'water' && (water_ruins_count < water_ruins_number || terrain !== 'ocean')) {
                map[ruin].above = 'ruin'; // actually there can be both ruin and resource on a single tile but only ruin is displayed; as it is just a map generator it doesn't matter
                if (terrain === 'ocean') {
                    water_ruins_count++;
                }
                for (let cell of circle(ruin, 1)) {
                    village_map[cell] = Math.max(village_map[cell], 2); // we won't use this array anymore anyway
                }
                ruins_count++;
            }
        }
        console.timeEnd('Ruin generation');
    }

    function check_resources(resource, capital) {
        let resources = 0;
        for (let neighbour of circle(capital, 1)) {
            if (map[neighbour]['above'] === resource) {
                resources++;
            }
        }
        return resources;
    }

    function post_generate(resource, underneath, quantity, capital) {
        let resources = check_resources(resource, capital);
        while (resources < quantity) {
            let pos = random_int(0, 8);
            let territory = circle(capital, 1);
            map[territory[pos]]['type'] = underneath;
            map[territory[pos]]['above'] = resource;
            for (let neighbour of plus_sign(territory[pos])) {
                if (map[neighbour]['type'] === 'ocean') {
                    map[neighbour]['type'] = 'water';
                }
            }
            resources = check_resources(resource, capital);
        }
    }

    // tribe specific things
    console.time('Tribe specific');
    for (let capital of capital_cells) {
        switch (map[capital]['tribe']) {
            case 'Imperius':
                post_generate('fruit', 'ground', 2, capital);
                break;
            case 'Bardur':
                post_generate('game', 'forest', 2, capital);
                break;
            case 'Kickoo':
                let resources = check_resources('fish', capital);
                while (resources < 2) {
                    let pos = random_int(0, 4);
                    let territory = plus_sign(capital);
                    map[territory[pos]]['type'] = 'water';
                    map[territory[pos]]['above'] = 'fish';
                    for (let neighbour of plus_sign(territory[pos])) {
                        if (map[neighbour]['type'] === 'water') {
                            map[neighbour]['type'] = 'ocean';
                            for (let double_neighbour of plus_sign(neighbour)) {
                                if (map[double_neighbour]['type'] !== 'water' && map[double_neighbour]['type'] !== 'ocean') {
                                    map[neighbour]['type'] = 'water';
                                    break;
                                }
                            }
                        }
                    }
                    resources = check_resources('fish', capital);
                }
                break;
            case 'Zebasi':
                post_generate('crop', 'ground', 1, capital);
                break;
            case 'Elyrion':
                post_generate('game', 'forest', 2, capital);
                break;
            case 'Polaris':
                for (let neighbour of circle(capital, 1)) {
                    map[neighbour]['tribe'] = 'Polaris';
                }
                break;
        }
    }
    console.timeEnd('Tribe specific');
    // we're done!

    console.time('Display');
    display_map(map);
    console.timeEnd('Display');
    console.log('_______________________');
    console.log('Number of villages: ' + village_count);
    console.log('Number of ruins: ' + ruins_number);
    console.log('_______________________');

    // display text-map if necessary
    let text_output_check = document.getElementById("text_output_check").checked;
    if (text_output_check)
        print_map(map);
    else
        document.getElementById("text_display").style.display='none';

    // 4개의 테스트 유닛 추가
    units = [];
    add_unit(5, 5, 'capital', 'Xin-xi', 1);
    add_unit(7, 7, 'village', 'Imperius', 2);
    add_unit(3, 8, 'village', 'Bardur', 3);
    add_unit(8, 3, 'village', 'Oumaji', 4);
}

// we use pythagorean distances
function distance(a, b, size) {
    let ax = a % size;
    let ay = a / size | 0;
    let bx = b % size;
    let by = b / size | 0;
    return Math.max(Math.abs(ax - bx), Math.abs(ay - by));
}

function print_map(map) {
    let seen_grid = Array(map_size**2 * 4);
    for (let i = 0; i < map_size**2 * 4; i++) {
        seen_grid[i] = '-';
    }
    for (let i = 0; i < map_size**2; i++) {
        let row = Math.floor(i / map_size);
        let column = i % map_size;
        seen_grid[map_size - 1 + column - row + (column + row) * map_size * 2] = map[row * map_size + column]['type'][0];
    }
    let output = '';
    for (let i = 0; i < map_size * 2; i++) {
        output += seen_grid.slice(i * map_size * 2, (i + 1) * map_size * 2).join('');
        output += '\n'
    }

    document.getElementById("text_display").innerText = output;
    document.getElementById("text_display").style.display='block';
}

function display_map(map) {
    if (!canvas) {
        if (!initializeCanvas()) return;
    }

    canvas.clearRect(0, 0, width, height);

    // 배경색 설정
    canvas.fillStyle = '#000000';
    canvas.fillRect(0, 0, width, height);

    let map_pixel_width = tile_size * map_size;
    let map_pixel_height = tile_size * map_size * 0.6;
    let x_offset = (width - map_pixel_width) / 2;
    let y_offset = (height - map_pixel_height) / 2;

    let tile_height = assets['Xin-xi']['ground'].height;
    let tile_width = assets['Xin-xi']['ground'].width;
    for (let i = 0; i < map_size**2; i++) {
        let row = i / map_size | 0;
        let column = i % map_size;
        let x = x_offset + (column - row) * tile_size / 2 + map_pixel_width / 2 - tile_size / 2;
        let y = y_offset + (column + row) * tile_size * 0.3;
        let type = map[row * map_size + column]['type'];
        let above = map[row * map_size + column]['above'];
        let tribe = map[row * map_size + column]['tribe'];
        if (general_terrain.includes(type)) {
            canvas.drawImage(assets[type], x, y, tile_size, assets[type].height * tile_size / assets[type].width);
        } else if (tribe) {
            if (type === 'forest' || type === 'mountain') {
                canvas.drawImage(assets[tribe]['ground'], x, y, tile_size, assets[tribe]['ground'].height * tile_size / assets[tribe]['ground'].width);
                let lowering = tribe === 'Kickoo' && type === 'mountain' ? 0.82 : 0.52;
                canvas.drawImage(assets[tribe][type], x, y + lowering * tile_size - tile_size * assets[tribe][type].height / assets[tribe][type].width, tile_size, assets[tribe]['ground'].height * tile_size / assets[tribe]['ground'].width);
            } else if (type === 'water' || type === 'ocean') {
                canvas.drawImage(assets[tribe][type], x, y - 0.3 * tile_size, tile_size, assets[tribe][type].height * tile_size / assets[tribe][type].width);
            } else {
                canvas.drawImage(assets[tribe][type], x, y, tile_size, assets[tribe][type].height * tile_size / assets[tribe][type].width);
            }
        }
        function draw_above(image) {
            canvas.drawImage(image, x, y, tile_size, image.height * tile_size / image.width);
        }
        if (above === 'capital') {
            canvas.drawImage(assets[tribe]['capital'], x, y - 0.3 * tile_size, tile_size, assets[tribe]['capital'].height * tile_size / assets[tribe]['capital'].width);
        } else if (above === 'whale') {
            draw_above(assets['whale']);
        } else if (above === 'village') {
            draw_above(assets['village']);
        } else if (above === 'game') {
            draw_above(assets[tribe]['game']);
        } else if (above === 'fruit') {
            draw_above(assets[tribe]['fruit']);
        } else if (above === 'crop') {
            draw_above(assets['crop']);
        } else if (above === 'fish') {
            draw_above(assets['fish']);
        } else if (above === 'metal') {
            draw_above(assets['metal']);
        } else if (above === 'ruin') {
            draw_above(assets['ruin']);
        }
    }

    // 유닛 그리기
    draw_units();
}

function random_int(min, max) {
    let rand = min + Math.random() * (max - min);
    return Math.floor(rand);
}

function rand_array_element(arr) {
    return arr[Math.random() * arr.length | 0];
}

function circle(center, radius) {
    let circle = [];
    let row = center / map_size | 0;
    let column = center % map_size;
    let i = row - radius;
    if (i >= 0 && i < map_size) {
        for (let j = column - radius; j < column + radius; j++) {
            if (j >= 0 && j < map_size) {
                circle.push(i * map_size + j)
            }
        }
    }
    i = row + radius;
    if (i >= 0 && i < map_size) {
        for (let j = column + radius; j > column - radius; j--) {
            if (j >= 0 && j < map_size) {
                circle.push(i * map_size + j)
            }
        }
    }
    let j = column - radius;
    if (j >= 0 && j < map_size) {
        for (let i = row + radius; i > row - radius; i--) {
            if (i >= 0 && i < map_size) {
                circle.push(i * map_size + j)
            }
        }
    }
    j = column + radius;
    if (j >= 0 && j < map_size) {
        for (let i = row - radius; i < row + radius; i++) {
            if (i >= 0 && i < map_size) {
                circle.push(i * map_size + j)
            }
        }
    }
    return circle;
}

function round(center, radius) {
    let round = [];
    for (let r = 1; r <= radius; r++) {
        round = round.concat(circle(center, r));
    }
    round.push(center);
    return round;
}

function plus_sign(center) {
    let plus_sign = [];
    let row = center / map_size | 0;
    let column = center % map_size;
    if (column > 0) {
        plus_sign.push(center - 1);
    }
    if (column < map_size - 1) {
        plus_sign.push(center + 1);
    }
    if (row > 0) {
        plus_sign.push(center - map_size);
    }
    if (row < map_size - 1) {
        plus_sign.push(center + map_size);
    }
    return plus_sign;
}

let selected_unit_id = null;
let units = [];
let move_count = 100;

function add_unit(x, y, type, tribe, id) {
    units.push({
        id: id,
        x: x,
        y: y,
        type: type,
        tribe: tribe,
        moves_left: move_count
    });
}

function get_tile_at_position(x, y) {
    let row = Math.floor(y / (tile_size * 0.6));
    let col = Math.floor((x - (width - tile_size * map_size) / 2) / tile_size);
    return row * map_size + col;
}

function draw_units() {
    for (let unit of units) {
        let x = (unit.x - unit.y) * tile_size / 2 + width / 2;
        let y = (unit.x + unit.y) * tile_size * 0.3;
        
        // 유닛 이미지 그리기
        let unit_image = assets[unit.tribe][unit.type];
        if (unit_image) {
            canvas.drawImage(unit_image, x - tile_size/2, y - tile_size/2, tile_size, tile_size);
        }
        
        // 유닛 ID 표시
        canvas.fillStyle = '#ffffff';
        canvas.font = '20px Arial';
        canvas.fillText(unit.id.toString(), x, y);
        
        // 남은 이동 횟수 표시
        canvas.fillStyle = '#00ff00';
        canvas.font = '12px Arial';
        canvas.fillText(unit.moves_left.toString(), x, y + 20);
    }
}

function move_unit(unit_id, direction) {
    const unit = units.find(u => u.id === unit_id);
    if (!unit || unit.moves_left <= 0) return false;
    
    const directions = {
        'up': {dx: 0, dy: -1},
        'down': {dx: 0, dy: 1},
        'left': {dx: -1, dy: 0},
        'right': {dx: 1, dy: 0},
        'up-left': {dx: -1, dy: -1},
        'up-right': {dx: 1, dy: -1},
        'down-left': {dx: -1, dy: 1},
        'down-right': {dx: 1, dy: 1}
    };
    
    const dir = directions[direction];
    if (!dir) return false;
    
    const new_x = unit.x + dir.dx;
    const new_y = unit.y + dir.dy;
    
    // 맵 경계 체크
    if (new_x < 0 || new_x >= map_size || new_y < 0 || new_y >= map_size) return false;
    
    // 다른 유닛과의 충돌 체크
    if (units.some(u => u.x === new_x && u.y === new_y)) return false;
    
    unit.x = new_x;
    unit.y = new_y;
    unit.moves_left--;
    
    // 맵을 다시 그리지 않고 유닛만 업데이트
    if (canvas) {
        display_map(map);
    }
    return true;
}

// 전역 함수로 노출
window.move_unit = move_unit;

// 클릭 기반 캐릭터(머리) 추가/이동 기능
const graphic_display = document.getElementById("graphic_display");
if (graphic_display) {
    graphic_display.addEventListener('click', function(e) {
        // 클릭 좌표를 맵 좌표로 변환
        const rect = graphic_display.getBoundingClientRect();
        const click_x = e.clientX - rect.left;
        const click_y = e.clientY - rect.top;
        // 맵 좌표 계산
        let row = Math.floor((click_y - (height - tile_size * map_size * 0.6) / 2) / (tile_size * 0.3) / 2);
        let col = Math.floor((click_x - (width - tile_size * map_size) / 2) / tile_size);
        if (row < 0 || col < 0 || row >= map_size || col >= map_size) return;
        // 이미 유닛이 있는지 확인
        const unit = units.find(u => u.x === col && u.y === row);
        if (selected_unit_id && !unit) {
            // 선택된 유닛이 있고, 빈 칸 클릭: 이동
            const sel_unit = units.find(u => u.id === selected_unit_id);
            if (sel_unit) {
                sel_unit.x = col;
                sel_unit.y = row;
                display_map(map);
            }
            selected_unit_id = null;
        } else if (unit) {
            // 유닛 클릭: 선택
            selected_unit_id = unit.id;
        } else {
            // 빈 칸 클릭: 새 캐릭터(머리) 추가 (임의로 Imperius, id=units.length+1)
            add_unit(col, row, 'capital', 'Imperius', units.length + 1);
            display_map(map);
        }
    });
}
