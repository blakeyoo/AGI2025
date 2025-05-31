// Evolutionary MLP-based Othello Agent

const N = 50;
const T = 50;
const windowSize = 5
const hiddenSize = 16;
const inputSize = windowSize * windowSize;
const mu = 0.05;
const p_crossover = 0.7;

// Othello 1판 (흑/백 지정)
function simulateGame(agent1, agent2, initialBoard, api) {
    let board;
    if (typeof initialBoard !== 'undefined' && initialBoard !== null) {
        board = JSON.parse(JSON.stringify(initialBoard));
    } else {
        throw new Error("No initial board is provided.");
    }

    let player = 1;
    while (true) {
        const validMoves = api.getValidMoves(board, player);
        let move = null;
        if (validMoves.length > 0) {
            move = (player === 1 ? agent1 : agent2)(board, player, validMoves, null);
        }
        if (move) {
            const result = api.simulateMove(board, player, move.row, move.col);
            if (result.valid) {
                board = result.resultingBoard;
            }
        }
        const blackHasMoves = api.getValidMoves(board, 1).length > 0;
        const whiteHasMoves = api.getValidMoves(board, 2).length > 0;
        if (!blackHasMoves && !whiteHasMoves) break;
        player = player === 1 ? 2 : 1;
    }

    let blackCount = 0, whiteCount = 0;
    for (let r = 0; r < board.length; r++) {
        for (let c = 0; c < board[r].length; c++) {
            if (board[r][c] === 1) blackCount++;
            else if (board[r][c] === 2) whiteCount++;
        }
    }

    let winner = 0;
    if (blackCount > whiteCount) winner = 1;
    else if (whiteCount > blackCount) winner = 2;

    return {
        winner: winner,
        diff: Math.abs(blackCount - whiteCount)
    };
}

function simulateMatch(a1, a2, initialBoard, api) {
    const result1 = simulateGame(a1, a2, initialBoard, api);
    const result2 = simulateGame(a2, a1, initialBoard, api);

    const score1 = result1.diff;
    const score2 = result2.diff;
    // a1이 이긴 경우
    if (result1.winner !== result2.winner) {
        return result1.winner === 1 ? a1 : a2;
    } else if (score1 > score2) {
        return a1;
    } else if (score2 > score1) {
        return a2;
    } else {
        return Math.random() < 0.5 ? a1 : a2;
    }
}


function sigmoid(x) {
    if (x >= 0) {
        const z = Math.exp(-x);
        return 1 / (1 + z);
    } else {
        const z = Math.exp(x);
        return z / (1 + z);
    }
}

function randomNormal() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function xavierNormal(nIn, nOut) {
    // 표준편차 sqrt(2 / (nIn + nOut))
    return randomNormal() * Math.sqrt(2 / (nIn + nOut));
}


// agent 생성
function makePositionalAgent(posTable) {
    return function(board, player, validMoves, makeMove) {
        if (validMoves.length === 0) return null;

        // Step 1: Compute values for each valid move
        const moveValues = validMoves.map(move => ({
            move,
            value: posTable[move.row][move.col]
        }));

        // Step 2: Find the max value
        const maxValue = Math.max(...moveValues.map(mv => mv.value));

        // Step 3: Collect all moves with max value
        const bestMoves = moveValues.filter(mv => mv.value === maxValue);

        // Step 4: Randomly choose one among them
        const randomIndex = Math.floor(Math.random() * bestMoves.length);
        return bestMoves[randomIndex].move;
    };
}


class LocalMLP {
    constructor(inputSize = 25, hiddenSize = 16) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        this.W1 = Array.from({ length: hiddenSize }, () =>
            Array.from({ length: inputSize }, () => randomNormal() * 0.5)
        );
        this.b1 = Array.from({ length: hiddenSize }, () => randomNormal() * 0.1);
        this.W2 = Array.from({ length: hiddenSize }, () => randomNormal() * 0.5);
        this.b2 = randomNormal() * 0.1;
    }

    forward(x) {
        const size = 5;

        // Reshape 25-length array into 5×5 matrix
        const mat = [];
        for (let i = 0; i < size; i++) {
            mat.push(x.slice(i * size, (i + 1) * size));
        }

        // Rotate 90° clockwise
        const rotate90 = m => {
            const N = m.length;
            return Array.from({ length: N }, (_, i) =>
                Array.from({ length: N }, (_, j) => m[N - j - 1][i])
            );
        };

        // Flip horizontally
        const flipH = m => m.map(row => [...row].reverse());

        // Flatten matrix
        const flatten = m => m.flat();

        // Generate all 8 symmetric versions
        const variants = [];
        let current = mat;
        for (let i = 0; i < 4; i++) {
            variants.push(flatten(current));
            variants.push(flatten(flipH(current)));
            current = rotate90(current);
        }

        // Compute average output over all variants
        let total = 0;
        for (const variant of variants) {
            const h = this.W1.map((row, i) =>
                Math.tanh(row.reduce((sum, w, j) => sum + w * variant[j], this.b1[i]))
            );
            const out = Math.tanh(this.W2.reduce((sum, w, i) => sum + w * h[i], this.b2));
            total += out;
        }

        return total / variants.length;
    }


    getParams() {
        // 1차 벡터로 반환
        return [
            ...this.W1.flat(),
            ...this.b1,
            ...this.W2,
            this.b2
        ];
    }

    setParams(params) {
        const hs = this.hiddenSize;
        const isz = this.inputSize;
        const sz = hs * isz;

        this.W1 = [];
        for (let i = 0; i < hs; i++) {
            this.W1.push(params.slice(i * isz, (i + 1) * isz));
        }

        this.b1 = params.slice(sz, sz + hs);
        this.W2 = params.slice(sz + hs, sz + hs + hs);
        this.b2 = params[params.length - 1];
    }
}

function extractWindow(board, x, y, size = windowSize, obstacleVal = 1) {
    const N = board.length;
    const half = Math.floor(size / 2);
    const window = [];

    for (let dx = -half; dx <= half; dx++) {
        for (let dy = -half; dy <= half; dy++) {
            const nx = x + dx, ny = y + dy;
            if (0 <= nx && nx < N && 0 <= ny && ny < N) {
                window.push(board[nx][ny] === 3 ? obstacleVal : 0);  // 장애물 위치만 1
            } else {
                window.push(obstacleVal);  // 보드 밖은 무조건 장애물
            }
        }
    }

    return window;  // 1D array, length = size * size
}

function computePositionalMap(board, mlp) {
    const N = board.length;
    const posMap = Array.from({ length: N }, () => Array(N).fill(0));

    for (let x = 0; x < N; x++) {
        for (let y = 0; y < N; y++) {
            if (board[x][y] === 3) {
                posMap[x][y] = 0;  // 장애물은 0
            } else {
                const input = extractWindow(board, x, y);
                posMap[x][y] = mlp.forward(input);
            }
        }
    }

    return posMap;
}

function dampenCenterBias(map, strength = 0.3) {
    const N = map.length;
    const center = (N - 1) / 2;
    const maxDist = Math.max(center, center); // 대각선 기준이 아니라, x/y 중 큰 쪽

    return map.map((row, r) =>
        row.map((val, c) => {
            const dist = Math.max(Math.abs(r - center), Math.abs(c - center));
            const dampFactor = 1 - strength * (1 - dist / maxDist); // 중심: 1-strength, 외곽: 1
            return val * dampFactor;
        })
    );
}



function makeMinimaxAgent(posTable, depth = 2) {
    return function(board, player, validMoves, makeMove, api) {
        if (validMoves.length === 0) return null;

        function evaluate(board, posTable) {
            let score = 0;
            for (let r = 0; r < board.length; r++) {
                for (let c = 0; c < board[r].length; c++) {
                    const val = board[r][c];
                    if (val === player) score += posTable[r][c];
                    else if (val === 3 - player) score -= posTable[r][c];
                }
            }
            return score;
        }

        function minimax(board, currentPlayer, currentDepth, alpha, beta) {
            if (currentDepth === 0) {
                return evaluate(board, posTable);
            }

            const moves = api.getValidMoves(board, currentPlayer);
            if (moves.length === 0) {
                // 상대도 수가 없으면 게임 종료
                const otherMoves = api.getValidMoves(board, 3 - currentPlayer);
                if (otherMoves.length === 0) {
                    return evaluate(board, posTable);
                }
                // 수 없으면 턴 넘김
                return minimax(board, 3 - currentPlayer, currentDepth - 1, alpha, beta);
            }

            if (currentPlayer === player) {
                let maxEval = -Infinity;
                for (const move of moves) {
                    const result = api.simulateMove(board, currentPlayer, move.row, move.col);
                    if (!result.valid) continue;
                    const eval = minimax(result.resultingBoard, 3 - currentPlayer, currentDepth - 1, alpha, beta);
                    maxEval = Math.max(maxEval, eval);
                    alpha = Math.max(alpha, eval);
                    if (beta <= alpha) break;
                }
                return maxEval;
            } else {
                let minEval = Infinity;
                for (const move of moves) {
                    const result = api.simulateMove(board, currentPlayer, move.row, move.col);
                    if (!result.valid) continue;
                    const eval = minimax(result.resultingBoard, 3 - currentPlayer, currentDepth - 1, alpha, beta);
                    minEval = Math.min(minEval, eval);
                    beta = Math.min(beta, eval);
                    if (beta <= alpha) break;
                }
                return minEval;
            }
        }

        // 루트에서 best move 선택
        let bestMove = null;
        let bestValue = -Infinity;
        for (const move of validMoves) {
            const result = api.simulateMove(board, player, move.row, move.col);
            if (!result.valid) continue;

            const value = minimax(result.resultingBoard, 3 - player, depth - 1, -Infinity, Infinity);
            if (value > bestValue) {
                bestValue = value;
                bestMove = move;
            }
        }

        return bestMove;
    };
}


function analyzeStage(stageConfig, initialBoard, validMoves, api) {
    const size = stageConfig.boardSize;

    const mlps = Array.from({ length: 2 * N }, () => new LocalMLP(inputSize, hiddenSize));

    const start = performance.now();
    for (let generation = 0; generation < T; generation++) {
        // Selection
        mlps.sort(() => Math.random() - 0.5);
        const positionalMaps = mlps.map(mlp => computePositionalMap(initialBoard, mlp));
        const params = mlps.map(mlp => mlp.getParams());

        const survived = [];
        for (let i = 0; i < N; i++) {
            const a1 = makePositionalAgent(positionalMaps[2 * i]);
            const a2 = makePositionalAgent(positionalMaps[2 * i + 1]);
            const result1 = simulateGame(a1, a2, initialBoard, api);
            const result2 = simulateGame(a2, a1, initialBoard, api);

            const score1 = result1.diff;
            const score2 = result2.diff;
            if (result1.winner !== result2.winner) {
                survived.push(result1.winner === 1 ? params[2 * i] : params[2 * i + 1]);
            } else if (score1 > score2) {
                survived.push(params[2 * i]);
            } else if (score2 > score1) {
                survived.push(params[2 * i + 1]);
            } else {
                survived.push(Math.random() < 0.5 ? params[2 * i] : params[2 * i + 1]);
            }
        }

        let crossovered = survived.concat(survived);
        crossovered.sort(() => Math.random() - 0.5);
        for (let i = 0; i < N; i++) {
            if (Math.random() < p_crossover) {
                const alpha = Math.random();
                const p1 = crossovered[2 * i];
                const p2 = crossovered[2 * i + 1];
                const new1 = p1.map((v, j) => alpha * v + (1 - alpha) * p2[j]);
                const new2 = p1.map((v, j) => (1 - alpha) * v + alpha * p2[j]);
                crossovered[2 * i] = new1;
                crossovered[2 * i + 1] = new2;
            }
        }

        const mutated = crossovered.map(p => p.map(v => v + randomNormal() * mu));
        for (let i = 0; i < 2 * N; i++) {
            mlps[i].setParams(mutated[i]);
        }

        if (performance.now() - start > 50000) {
            console.log('Time limit exceeded. Stopping evolution.');
            break;
        }
    }

    // 평균 positional map 계산 전 normalize (장애물/초기돌 무시)
    const finalMaps = mlps.map(mlp => computePositionalMap(initialBoard, mlp)).map(map => {
        let min = Infinity, max = -Infinity;
        for (let r = 0; r < map.length; r++) {
            for (let c = 0; c < map[r].length; c++) {
                // 장애물(3), 초기 돌(1,2)은 무시
                if (initialBoard[r][c] === 3 || initialBoard[r][c] === 1 || initialBoard[r][c] === 2) continue;
                if (map[r][c] < min) min = map[r][c];
                if (map[r][c] > max) max = map[r][c];
            }
        }
        const range = max - min;
        return map.map((row, r) => row.map((v, c) => {
            // 장애물(3), 초기 돌(1,2) → 무조건 0
            if (initialBoard[r][c] === 3 || initialBoard[r][c] === 1 || initialBoard[r][c] === 2) return 0;
            return range === 0 ? 0 : -1 + 2 * (v - min) / range;
        }));
    });

    // 1. finalMaps의 각 위치별 평균값 계산
    const Nmap = finalMaps.length;
    const sizeY = finalMaps[0].length;
    const sizeX = finalMaps[0][0].length;

    const meanMap = Array.from({ length: sizeY }, (_, y) =>
        Array.from({ length: sizeX }, (_, x) => {
            let sum = 0, count = 0;
            for (let i = 0; i < Nmap; i++) {
                sum += finalMaps[i][y][x];
                count++;
            }
            return count === 0 ? 0 : sum / count;
        })
    );

    // 2. 평균 map을 기반으로 positional agent 생성
    const damepnMap = dampenCenterBias(meanMap, strength = 0.8)
    damepnMap.forEach((row, i) => console.log('pos row', i, ':', row.map(v => v.toFixed(3)).join(' ')));
    return makePositionalAgent(damepnMap);

    
    // 2. 각 positional map으로 agent 생성
    // const agents = finalMaps.map(map => makePositionalAgent(map));

    // // 3. 토너먼트(리그식)로 최종 승자 agent 선정
    // let winnerIdx = 0;
    // for (let i = 1; i < agents.length; i++) {
    //     const agentA = agents[winnerIdx];
    //     const agentB = agents[i];
    //     const matchWinner = simulateMatch(agentA, agentB, initialBoard, api); // 1 또는 2 반환
    //     if (matchWinner === 2) winnerIdx = i;
    // }

    // 4. 최종 승자 agent 반환
    // finalMaps[winnerIdx].forEach((row, i) => console.log('pos row', i, ':', row.map(v => v.toFixed(3)).join(' ')));
    // return makePositionalAgent(dampenCenterBias(finalMaps[winnerIdx]));

}
