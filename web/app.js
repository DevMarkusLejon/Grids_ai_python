(function () {
  const Side = {
    BLUE: "blue",
    RED: "red",
    other(side) {
      return side === this.BLUE ? this.RED : this.BLUE;
    },
    short(side) {
      return side === this.BLUE ? "B" : "R";
    },
  };

  const AttackRole = {
    MELEE: "melee",
    RANGED: "ranged",
    HEALER: "healer",
  };

  const CardKind = {
    UNIT: "unit",
    ITEM: "item",
  };

  const UNIT_BLUEPRINTS = {
    commander: { key: "commander", name: "Commander", glyph: "C", maxHp: 300, damage: 20, moveRange: 2, attackRange: 1, deployCost: 0, role: AttackRole.MELEE, canDeploy: false },
    warrior: { key: "warrior", name: "Warrior", glyph: "W", maxHp: 100, damage: 40, moveRange: 2, attackRange: 1, deployCost: 2, role: AttackRole.MELEE, canDeploy: true },
    archer: { key: "archer", name: "Archer", glyph: "A", maxHp: 80, damage: 20, moveRange: 2, attackRange: 4, deployCost: 2, role: AttackRole.RANGED, canDeploy: true },
    healer: { key: "healer", name: "Healer", glyph: "H", maxHp: 80, damage: 30, moveRange: 3, attackRange: 3, deployCost: 2, role: AttackRole.HEALER, canDeploy: true },
    assassin: { key: "assassin", name: "Assassin", glyph: "S", maxHp: 70, damage: 35, moveRange: 3, attackRange: 1, deployCost: 1, role: AttackRole.MELEE, canDeploy: true },
    viking: { key: "viking", name: "Viking", glyph: "V", maxHp: 110, damage: 30, moveRange: 2, attackRange: 1, deployCost: 2, role: AttackRole.MELEE, canDeploy: true },
  };

  const ITEM_BLUEPRINTS = {
    fireball: { key: "fireball", name: "Fireball", cost: 1, effect: "damage_enemy", power: 30, description: "Deal 30 damage to an enemy unit anywhere on the board." },
    strength_tonic: { key: "strength_tonic", name: "Strength Tonic", cost: 2, effect: "buff_friendly_damage", power: 10, description: "Give a friendly unit +10 permanent damage." },
  };

  const ITEM_ASSETS = {
    fireball: {
      iconSrc: "./assets/items/icons/fireball.png",
      effectSrc: "./assets/items/effects/fireball.png",
      effectFrameRatio: 0.6,
    },
    strength_tonic: {
      iconSrc: "./assets/items/icons/strength-tonic.png",
      effectSrc: "./assets/items/effects/strength-tonic.png",
      effectFrameRatio: 0.6,
    },
  };

  const MAPS = {
    plains: {
      name: "Plains",
      width: 10,
      height: 7,
      blockers: new Set(["6,1", "3,5", "5,2", "4,4"]),
      blueCommander: [1, 3],
      redCommander: [8, 3],
      blueDeploy: new Set(Array.from({ length: 7 }, (_, y) => `0,${y}`)),
      redDeploy: new Set(Array.from({ length: 7 }, (_, y) => `9,${y}`)),
    },
  };

  const DEFAULT_UNIT_DECK = ["warrior", "archer", "warrior", "archer", "viking", "assassin", "viking", "assassin", "healer", "healer"];
  const DEFAULT_ITEM_DECK = ["fireball", "strength_tonic", "fireball", "strength_tonic"];

  const CONFIG = {
    maxActions: 7,
    maxHandSize: 7,
    startingHandSize: 5,
    drawCostUnit: 1,
    drawCostItem: 1,
    maxHalfTurns: 80,
    aiSearchWidth: 5,
    aiSearchDepth: 8,
    neuralScale: 120,
    heuristicScale: 1,
    neuralSearchWidth: 3,
    neuralSearchDepth: 4,
    aiDelays: [720, 360, 140],
  };

  const STRONGEST_VALUE_MODEL = window.GRIDS_STRONGEST_VALUE_MODEL || null;
  const UNIT_KEYS = ["commander", "warrior", "archer", "healer", "assassin", "viking"];

  const DEFAULT_WEIGHTS = {
    bias: -0.7917604190523033,
    enemyCommanderDelta: 12.774282754309496,
    ownCommanderDelta: -21.48864176598216,
    enemyUnitDelta: 47.81908068985808,
    ownUnitDelta: -53.45474315105486,
    enemyTotalHpDelta: 6.021867223547596,
    ownTotalHpDelta: -3.26455564572877,
    enemyUnitValueDelta: 0.2,
    ownUnitValueDelta: -0.25,
    forwardPressureDelta: 4.540340913863352,
    commanderDistanceDelta: 5,
    enemyCommanderThreatDelta: 2.2,
    ownCommanderThreatDelta: -2.8,
    lethalThreat: 180,
    ownLethalRisk: -220,
    moveEnablesAttack: 16,
    effectiveHealing: 1.6,
    overkillDamage: -0.35,
    handDelta: 4.869740324800604,
    deploy: 5.089783031748675,
    move: 6.987327891951255,
    attack: 0.3039665044590707,
    heal: -2.649448948095257,
    item: 4.64289201489151,
    drawUnit: 6.034211969542737,
    drawItem: 2.3666803449055154,
    endTurn: -4.267859011069304,
    remainingAp: -0.33929613525688956,
    win: 10000,
    loss: -10000,
  };

  const SPRITE_SHEETS = {
    blue: {
      commander: { src: "./assets/sprites/blue/commander.png", frameRatio: 0.6 },
      warrior: { src: "./assets/sprites/blue/warrior.png", frameRatio: 0.5 },
      archer: { src: "./assets/sprites/blue/archer.png", frameRatio: 0.5 },
      healer: { src: "./assets/sprites/blue/healer.png", frameRatio: 0.6 },
      assassin: { src: "./assets/sprites/blue/assassin.png", frameRatio: 0.6 },
      viking: { src: "./assets/sprites/blue/viking.png", frameRatio: 0.547 },
    },
    red: {
      commander: { src: "./assets/sprites/red/commander.png", frameRatio: 0.5 },
      warrior: { src: "./assets/sprites/red/warrior.png", frameRatio: 0.5 },
      archer: { src: "./assets/sprites/red/archer.png", frameRatio: 0.6 },
      healer: { src: "./assets/sprites/red/healer.png", frameRatio: 0.5 },
      assassin: { src: "./assets/sprites/red/assassin.png", frameRatio: 0.6 },
      viking: { src: "./assets/sprites/red/viking.png", frameRatio: 0.5 },
    },
  };

  function coordKey(x, y) {
    return `${x},${y}`;
  }

  function parseCoord(key) {
    return key.split(",").map(Number);
  }

  function shuffle(list) {
    const copy = [...list];
    for (let i = copy.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
  }

  function drawStartingHand(unitDeck, itemDeck) {
    const hand = [];
    while (hand.length < CONFIG.startingHandSize && (unitDeck.length || itemDeck.length)) {
      const takeUnit = unitDeck.length > 0 && (itemDeck.length === 0 || Math.random() < 0.5);
      if (takeUnit) {
        hand.push({ kind: CardKind.UNIT, key: unitDeck.shift() });
      } else {
        hand.push({ kind: CardKind.ITEM, key: itemDeck.shift() });
      }
    }
    return hand;
  }

  function createUnit(blueprintKey, side, x, y, unitId) {
    const blueprint = UNIT_BLUEPRINTS[blueprintKey];
    return {
      unitId,
      blueprintKey,
      side,
      x,
      y,
      hp: blueprint.maxHp,
      damageBonus: 0,
      attackedThisTurn: false,
    };
  }

  function blueprintForUnit(unit) {
    return UNIT_BLUEPRINTS[unit.blueprintKey];
  }

  function unitDamage(unit) {
    return blueprintForUnit(unit).damage + unit.damageBonus;
  }

  function unitMaxHp(unit) {
    return blueprintForUnit(unit).maxHp;
  }

  function unitName(unit) {
    return blueprintForUnit(unit).name;
  }

  function unitRole(unit) {
    return blueprintForUnit(unit).role;
  }

  function isCommander(unit) {
    return unit.blueprintKey === "commander";
  }

  function createInitialState() {
    const map = MAPS.plains;
    const blueUnitDeck = shuffle(DEFAULT_UNIT_DECK);
    const redUnitDeck = shuffle(DEFAULT_UNIT_DECK);
    const blueItemDeck = shuffle(DEFAULT_ITEM_DECK);
    const redItemDeck = shuffle(DEFAULT_ITEM_DECK);

    const state = {
      map,
      currentSide: Side.BLUE,
      turnNumber: 1,
      actionsLeft: CONFIG.maxActions,
      hands: {
        [Side.BLUE]: drawStartingHand(blueUnitDeck, blueItemDeck),
        [Side.RED]: drawStartingHand(redUnitDeck, redItemDeck),
      },
      unitDecks: {
        [Side.BLUE]: blueUnitDeck,
        [Side.RED]: redUnitDeck,
      },
      itemDecks: {
        [Side.BLUE]: blueItemDeck,
        [Side.RED]: redItemDeck,
      },
      units: [],
      nextUnitId: 3,
      winner: null,
      winnerReason: "",
      halfTurnsPlayed: 0,
      log: ["Started a new Plains match.", "Both commanders are on the field."],
    };

    state.units.push(createUnit("commander", Side.BLUE, map.blueCommander[0], map.blueCommander[1], 1));
    state.units.push(createUnit("commander", Side.RED, map.redCommander[0], map.redCommander[1], 2));
    return state;
  }

  function cloneState(state) {
    return {
      map: state.map,
      currentSide: state.currentSide,
      turnNumber: state.turnNumber,
      actionsLeft: state.actionsLeft,
      hands: {
        [Side.BLUE]: state.hands[Side.BLUE].map((card) => ({ ...card })),
        [Side.RED]: state.hands[Side.RED].map((card) => ({ ...card })),
      },
      unitDecks: {
        [Side.BLUE]: [...state.unitDecks[Side.BLUE]],
        [Side.RED]: [...state.unitDecks[Side.RED]],
      },
      itemDecks: {
        [Side.BLUE]: [...state.itemDecks[Side.BLUE]],
        [Side.RED]: [...state.itemDecks[Side.RED]],
      },
      units: state.units.map((unit) => ({ ...unit })),
      nextUnitId: state.nextUnitId,
      winner: state.winner,
      winnerReason: state.winnerReason,
      halfTurnsPlayed: state.halfTurnsPlayed,
      log: [...state.log],
    };
  }

  function unitAt(state, x, y) {
    return state.units.find((unit) => unit.x === x && unit.y === y) || null;
  }

  function unitsForSide(state, side) {
    return state.units
      .filter((unit) => unit.side === side)
      .sort((a, b) => a.y - b.y || a.x - b.x || a.unitId - b.unitId);
  }

  function findCommander(state, side) {
    return unitsForSide(state, side).find((unit) => isCommander(unit)) || null;
  }

  function commanderHp(state, side) {
    const commander = findCommander(state, side);
    return commander ? commander.hp : 0;
  }

  function commanderMaxHp(state, side) {
    const commander = findCommander(state, side);
    return commander ? unitMaxHp(commander) : UNIT_BLUEPRINTS.commander.maxHp;
  }

  function totalHp(state, side) {
    return unitsForSide(state, side).reduce((sum, unit) => sum + unit.hp, 0);
  }

  function distance(a, b) {
    return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
  }

  function attackInRange(attacker, target) {
    const blueprint = blueprintForUnit(attacker);
    const dx = Math.abs(attacker.x - target.x);
    const dy = Math.abs(attacker.y - target.y);

    if (blueprint.attackRange === 1) {
      return (dx !== 0 || dy !== 0) && dx <= 1 && dy <= 1;
    }

    return distance([attacker.x, attacker.y], [target.x, target.y]) <= blueprint.attackRange;
  }

  function inBounds(state, x, y) {
    return x >= 0 && x < state.map.width && y >= 0 && y < state.map.height;
  }

  function isBlocked(state, x, y) {
    return state.map.blockers.has(coordKey(x, y));
  }

  function neighbours(x, y) {
    return [
      [x + 1, y],
      [x - 1, y],
      [x, y + 1],
      [x, y - 1],
    ];
  }

  function stepDirection(value) {
    if (value === 0) {
      return 0;
    }
    return value > 0 ? 1 : -1;
  }

  function openCell(state, x, y) {
    return inBounds(state, x, y) && !isBlocked(state, x, y) && !unitAt(state, x, y);
  }

  function knockbackDestination(state, attacker, target) {
    const dx = target.x - attacker.x;
    const dy = target.y - attacker.y;
    const candidates = [];

    if (dx !== 0 && dy !== 0) {
      candidates.push([target.x + stepDirection(dx), target.y + stepDirection(dy)]);
    }

    if (Math.abs(dx) >= Math.abs(dy) && dx !== 0) {
      candidates.push([target.x + stepDirection(dx), target.y]);
      if (dy !== 0) {
        candidates.push([target.x, target.y + stepDirection(dy)]);
      }
    } else if (dy !== 0) {
      candidates.push([target.x, target.y + stepDirection(dy)]);
      if (dx !== 0) {
        candidates.push([target.x + stepDirection(dx), target.y]);
      }
    }

    return candidates.find(([x, y]) => openCell(state, x, y)) || null;
  }

  function availableDeployCells(state, side) {
    const zone = side === Side.BLUE ? state.map.blueDeploy : state.map.redDeploy;
    return [...zone]
      .map(parseCoord)
      .filter(([x, y]) => !isBlocked(state, x, y) && !unitAt(state, x, y));
  }

  function reachableCells(state, unit) {
    const frontier = [[unit.x, unit.y, 0]];
    const seen = new Set([coordKey(unit.x, unit.y)]);
    const reachable = [];
    const moveRange = blueprintForUnit(unit).moveRange;

    while (frontier.length > 0) {
      const [x, y, steps] = frontier.shift();
      if (steps === moveRange) {
        continue;
      }
      for (const [nx, ny] of neighbours(x, y)) {
        const key = coordKey(nx, ny);
        if (seen.has(key)) {
          continue;
        }
        seen.add(key);
        if (!inBounds(state, nx, ny) || isBlocked(state, nx, ny)) {
          continue;
        }
        const occupant = unitAt(state, nx, ny);
        if (occupant && occupant.unitId !== unit.unitId) {
          continue;
        }
        reachable.push([nx, ny]);
        frontier.push([nx, ny, steps + 1]);
      }
    }
    return reachable;
  }

  function sideScore(state, side) {
    const enemy = Side.other(side);
    let score = 0;
    score += commanderHp(state, side) * 5;
    score -= commanderHp(state, enemy) * 5;
    score += unitsForSide(state, side).length * 8;
    score -= unitsForSide(state, enemy).length * 8;
    score += totalHp(state, side) * 0.25;
    score -= totalHp(state, enemy) * 0.25;
    score += forwardPressure(state, side) * 2;
    score -= forwardPressure(state, enemy) * 2;
    return score;
  }

  function forwardPressure(state, side) {
    return unitsForSide(state, side).reduce((sum, unit) => {
      if (isCommander(unit)) {
        return sum;
      }
      const progress = side === Side.BLUE ? unit.x : state.map.width - 1 - unit.x;
      const centerBias = state.map.height / 2 - Math.abs(unit.y - (state.map.height - 1) / 2);
      return sum + progress + centerBias * 0.25;
    }, 0);
  }

  function unitValue(unit) {
    if (isCommander(unit)) {
      return 420 + unit.hp;
    }
    const blueprint = blueprintForUnit(unit);
    return unitMaxHp(unit) * 0.35 + unit.hp * 0.25 + unitDamage(unit) * 1.4 + blueprint.moveRange * 5 + blueprint.attackRange * 6;
  }

  function totalUnitValue(state, side) {
    return unitsForSide(state, side).reduce((sum, unit) => sum + unitValue(unit), 0);
  }

  function commanderDistance(state, side) {
    const enemyCommander = findCommander(state, Side.other(side));
    if (!enemyCommander) {
      return 0;
    }
    const distances = unitsForSide(state, side)
      .filter((unit) => !isCommander(unit))
      .map((unit) => distance([unit.x, unit.y], [enemyCommander.x, enemyCommander.y]));
    return distances.length ? Math.min(...distances) : state.map.width + state.map.height;
  }

  function commanderThreat(state, attackerSide, defenderSide) {
    const commander = findCommander(state, defenderSide);
    if (!commander) {
      return 0;
    }
    const attacks = [];
    for (const unit of unitsForSide(state, attackerSide)) {
      if (unitRole(unit) === AttackRole.HEALER || unit.attackedThisTurn) {
        continue;
      }
      if (attackInRange(unit, commander)) {
        attacks.push(unitDamage(unit));
      }
    }
    for (const card of state.hands[attackerSide]) {
      if (card.kind !== CardKind.ITEM) {
        continue;
      }
      const item = ITEM_BLUEPRINTS[card.key];
      if (item.effect === "damage_enemy" && state.actionsLeft >= item.cost) {
        attacks.push(item.power);
      }
    }
    return attacks.sort((a, b) => b - a).slice(0, Math.max(state.actionsLeft, 0)).reduce((sum, value) => sum + value, 0);
  }

  function moveEnablesAttack(before, after, action, player) {
    if (action.kind !== "move") {
      return false;
    }
    const beforeUnit = before.units.find((unit) => unit.unitId === action.unitId);
    const afterUnit = after.units.find((unit) => unit.unitId === action.unitId);
    if (!beforeUnit || !afterUnit || unitRole(beforeUnit) === AttackRole.HEALER || beforeUnit.attackedThisTurn) {
      return false;
    }
    return unitsForSide(after, Side.other(player)).some((enemy) => attackInRange(afterUnit, enemy));
  }

  function effectiveHealing(before, after, action) {
    if (action.kind !== "attack") {
      return 0;
    }
    const attacker = before.units.find((unit) => unit.unitId === action.unitId);
    if (!attacker || unitRole(attacker) !== AttackRole.HEALER) {
      return 0;
    }
    const beforeTarget = before.units.find((unit) => unit.unitId === action.targetUnitId);
    const afterTarget = after.units.find((unit) => unit.unitId === action.targetUnitId);
    if (!beforeTarget || !afterTarget) {
      return 0;
    }
    return Math.max(0, afterTarget.hp - beforeTarget.hp);
  }

  function overkillDamage(before, action) {
    const target = before.units.find((unit) => unit.unitId === action.targetUnitId);
    if (!target) {
      return 0;
    }
    let damage = 0;
    if (action.kind === "attack") {
      const attacker = before.units.find((unit) => unit.unitId === action.unitId);
      if (!attacker || unitRole(attacker) === AttackRole.HEALER) {
        return 0;
      }
      damage = unitDamage(attacker);
    } else if (action.kind === "play_item") {
      const card = before.hands[before.currentSide][action.handIndex];
      if (!card || card.kind !== CardKind.ITEM) {
        return 0;
      }
      const item = ITEM_BLUEPRINTS[card.key];
      if (item.effect !== "damage_enemy") {
        return 0;
      }
      damage = item.power;
    }
    return Math.max(0, damage - target.hp);
  }

  function legalActions(state) {
    if (state.winner) {
      return [];
    }

    const actions = [{ kind: "end_turn" }];
    const hand = state.hands[state.currentSide];

    if (state.actionsLeft >= CONFIG.drawCostUnit && hand.length < CONFIG.maxHandSize && state.unitDecks[state.currentSide].length) {
      actions.push({ kind: "draw_unit" });
    }
    if (state.actionsLeft >= CONFIG.drawCostItem && hand.length < CONFIG.maxHandSize && state.itemDecks[state.currentSide].length) {
      actions.push({ kind: "draw_item" });
    }

    hand.forEach((card, handIndex) => {
      if (card.kind === CardKind.UNIT) {
        const blueprint = UNIT_BLUEPRINTS[card.key];
        if (state.actionsLeft < blueprint.deployCost) {
          return;
        }
        for (const [x, y] of availableDeployCells(state, state.currentSide)) {
          actions.push({ kind: "deploy", handIndex, destination: [x, y] });
        }
      } else {
        const item = ITEM_BLUEPRINTS[card.key];
        if (state.actionsLeft < item.cost) {
          return;
        }
        if (item.effect === "damage_enemy") {
          for (const target of unitsForSide(state, Side.other(state.currentSide))) {
            actions.push({ kind: "play_item", handIndex, targetUnitId: target.unitId });
          }
        } else if (item.effect === "buff_friendly_damage") {
          for (const target of unitsForSide(state, state.currentSide)) {
            actions.push({ kind: "play_item", handIndex, targetUnitId: target.unitId });
          }
        }
      }
    });

    if (state.actionsLeft <= 0) {
      return actions;
    }

    for (const unit of unitsForSide(state, state.currentSide)) {
      for (const [x, y] of reachableCells(state, unit)) {
        actions.push({ kind: "move", unitId: unit.unitId, destination: [x, y] });
      }

      if (unit.attackedThisTurn) {
        continue;
      }

      if (unitRole(unit) === AttackRole.HEALER) {
        for (const target of unitsForSide(state, state.currentSide)) {
          if (target.unitId !== unit.unitId && target.hp < unitMaxHp(target) && attackInRange(unit, target)) {
            actions.push({ kind: "attack", unitId: unit.unitId, targetUnitId: target.unitId });
          }
        }
      } else {
        for (const target of unitsForSide(state, Side.other(state.currentSide))) {
          if (attackInRange(unit, target)) {
            actions.push({ kind: "attack", unitId: unit.unitId, targetUnitId: target.unitId });
          }
        }
      }
    }

    return actions;
  }

  function removeUnit(state, unitId) {
    const index = state.units.findIndex((unit) => unit.unitId === unitId);
    if (index === -1) {
      return;
    }
    const [unit] = state.units.splice(index, 1);
    state.log.push(`${unitName(unit)} was defeated.`);
    if (isCommander(unit)) {
      state.winner = Side.other(unit.side);
      state.winnerReason = `${unit.side} commander defeated`;
      state.log.push(`${Side.short(state.winner)} wins by defeating the enemy commander.`);
    }
  }

  function applyAction(state, action) {
    if (state.winner) {
      return state;
    }

    switch (action.kind) {
      case "draw_unit": {
        const cardKey = state.unitDecks[state.currentSide].shift();
        state.hands[state.currentSide].push({ kind: CardKind.UNIT, key: cardKey });
        state.actionsLeft -= CONFIG.drawCostUnit;
        state.log.push(`${Side.short(state.currentSide)} drew unit ${UNIT_BLUEPRINTS[cardKey].name}.`);
        break;
      }
      case "draw_item": {
        const cardKey = state.itemDecks[state.currentSide].shift();
        state.hands[state.currentSide].push({ kind: CardKind.ITEM, key: cardKey });
        state.actionsLeft -= CONFIG.drawCostItem;
        state.log.push(`${Side.short(state.currentSide)} drew item ${ITEM_BLUEPRINTS[cardKey].name}.`);
        break;
      }
      case "deploy": {
        const card = state.hands[state.currentSide][action.handIndex];
        const blueprint = UNIT_BLUEPRINTS[card.key];
        state.hands[state.currentSide].splice(action.handIndex, 1);
        state.actionsLeft -= blueprint.deployCost;
        state.units.push(createUnit(card.key, state.currentSide, action.destination[0], action.destination[1], state.nextUnitId));
        state.nextUnitId += 1;
        state.log.push(`${Side.short(state.currentSide)} deployed ${blueprint.name} at ${action.destination.join(",")}.`);
        break;
      }
      case "move": {
        const unit = state.units.find((entry) => entry.unitId === action.unitId);
        if (!unit) {
          return state;
        }
        const start = `${unit.x},${unit.y}`;
        unit.x = action.destination[0];
        unit.y = action.destination[1];
        state.actionsLeft -= 1;
        state.log.push(`${Side.short(state.currentSide)} moved ${unitName(unit)} from ${start} to ${action.destination.join(",")}.`);
        break;
      }
      case "attack": {
        const unit = state.units.find((entry) => entry.unitId === action.unitId);
        const target = state.units.find((entry) => entry.unitId === action.targetUnitId);
        if (!unit || !target) {
          return state;
        }
        state.actionsLeft -= 1;
        unit.attackedThisTurn = true;
        if (unitRole(unit) === AttackRole.HEALER) {
          const before = target.hp;
          target.hp = Math.min(unitMaxHp(target), target.hp + unitDamage(unit));
          state.log.push(`${Side.short(state.currentSide)} healed ${unitName(target)} for ${target.hp - before} using ${unitName(unit)}.`);
        } else {
          const damage = unitDamage(unit);
          target.hp -= damage;
          state.log.push(`${Side.short(state.currentSide)} attacked ${unitName(target)} with ${unitName(unit)} for ${damage}.`);
          if (target.hp <= 0) {
            removeUnit(state, target.unitId);
          } else {
            const knockback = knockbackDestination(state, unit, target);
            if (knockback) {
              const start = `${target.x},${target.y}`;
              target.x = knockback[0];
              target.y = knockback[1];
              state.log.push(`${unitName(target)} was knocked back from ${start} to ${knockback.join(",")}.`);
            }
          }
        }
        break;
      }
      case "play_item": {
        const card = state.hands[state.currentSide][action.handIndex];
        const item = ITEM_BLUEPRINTS[card.key];
        const target = state.units.find((entry) => entry.unitId === action.targetUnitId);
        if (!target) {
          return state;
        }
        state.hands[state.currentSide].splice(action.handIndex, 1);
        state.actionsLeft -= item.cost;
        if (item.effect === "damage_enemy") {
          target.hp -= item.power;
          state.log.push(`${Side.short(state.currentSide)} cast ${item.name} on ${unitName(target)} for ${item.power} damage.`);
          if (target.hp <= 0) {
            removeUnit(state, target.unitId);
          }
        } else if (item.effect === "buff_friendly_damage") {
          target.damageBonus += item.power;
          state.log.push(`${Side.short(state.currentSide)} used ${item.name} on ${unitName(target)}, gaining +${item.power} damage.`);
        }
        break;
      }
      case "end_turn": {
        state.currentSide = Side.other(state.currentSide);
        state.turnNumber += 1;
        state.halfTurnsPlayed += 1;
        state.actionsLeft = CONFIG.maxActions;
        for (const unit of unitsForSide(state, state.currentSide)) {
          unit.attackedThisTurn = false;
        }
        state.log.push(`It is now ${Side.short(state.currentSide)}'s turn.`);
        if (state.halfTurnsPlayed >= CONFIG.maxHalfTurns && !state.winner) {
          const blueScore = sideScore(state, Side.BLUE);
          const redScore = sideScore(state, Side.RED);
          state.winner = blueScore >= redScore ? Side.BLUE : Side.RED;
          state.winnerReason = "score advantage after turn limit";
          state.log.push(`${Side.short(state.winner)} wins on score after the turn limit.`);
        }
        break;
      }
      default:
        break;
    }

    return state;
  }

  function actionsEqual(a, b) {
    const aDest = a.destination ? a.destination.join(",") : "";
    const bDest = b.destination ? b.destination.join(",") : "";
    return a.kind === b.kind && a.handIndex === b.handIndex && a.unitId === b.unitId && a.targetUnitId === b.targetUnitId && aDest === bDest;
  }

  function features(before, after, action, player) {
    const enemy = Side.other(player);
    const beforeEnemyThreat = commanderThreat(before, player, enemy);
    const afterEnemyThreat = commanderThreat(after, player, enemy);
    const beforeOwnThreat = commanderThreat(before, enemy, player);
    const afterOwnThreat = commanderThreat(after, enemy, player);
    const afterEnemyCommanderHp = commanderHp(after, enemy);
    const afterOwnCommanderHp = commanderHp(after, player);
    return {
      bias: 1,
      enemyCommanderDelta: commanderHp(before, enemy) - commanderHp(after, enemy),
      ownCommanderDelta: commanderHp(before, player) - commanderHp(after, player),
      enemyUnitDelta: unitsForSide(before, enemy).length - unitsForSide(after, enemy).length,
      ownUnitDelta: unitsForSide(before, player).length - unitsForSide(after, player).length,
      enemyTotalHpDelta: totalHp(before, enemy) - totalHp(after, enemy),
      ownTotalHpDelta: totalHp(before, player) - totalHp(after, player),
      enemyUnitValueDelta: totalUnitValue(before, enemy) - totalUnitValue(after, enemy),
      ownUnitValueDelta: totalUnitValue(before, player) - totalUnitValue(after, player),
      forwardPressureDelta: forwardPressure(after, player) - forwardPressure(before, player),
      commanderDistanceDelta: commanderDistance(before, player) - commanderDistance(after, player),
      enemyCommanderThreatDelta: afterEnemyThreat - beforeEnemyThreat,
      ownCommanderThreatDelta: afterOwnThreat - beforeOwnThreat,
      lethalThreat: !after.winner && afterEnemyThreat >= afterEnemyCommanderHp && afterEnemyCommanderHp > 0 ? 1 : 0,
      ownLethalRisk: !after.winner && afterOwnThreat >= afterOwnCommanderHp && afterOwnCommanderHp > 0 ? 1 : 0,
      moveEnablesAttack: moveEnablesAttack(before, after, action, player) ? 1 : 0,
      effectiveHealing: effectiveHealing(before, after, action),
      overkillDamage: overkillDamage(before, action),
      handDelta: after.hands[player].length - before.hands[player].length,
      deploy: action.kind === "deploy" ? 1 : 0,
      move: action.kind === "move" ? 1 : 0,
      attack: action.kind === "attack" ? 1 : 0,
      heal: action.kind === "attack" && unitRole(before.units.find((unit) => unit.unitId === action.unitId)) === AttackRole.HEALER ? 1 : 0,
      item: action.kind === "play_item" ? 1 : 0,
      drawUnit: action.kind === "draw_unit" ? 1 : 0,
      drawItem: action.kind === "draw_item" ? 1 : 0,
      endTurn: action.kind === "end_turn" ? 1 : 0,
      remainingAp: after.actionsLeft,
    };
  }

  function scoreAction(before, after, action, player) {
    let score = 0;
    const featureValues = features(before, after, action, player);
    Object.entries(featureValues).forEach(([key, value]) => {
      score += (DEFAULT_WEIGHTS[key] || 0) * value;
    });
    if (after.winner) {
      score += after.winner === player ? DEFAULT_WEIGHTS.win : DEFAULT_WEIGHTS.loss;
    } else if (commanderHp(after, Side.other(player)) <= 40) {
      score += 25;
    }
    return score;
  }

  function cellIndex(state, x, y) {
    return y * state.map.width + x;
  }

  function unitKindValue(unit) {
    return (UNIT_KEYS.indexOf(unit.blueprintKey) + 1) / UNIT_KEYS.length;
  }

  function setEncodedCell(values, state, channel, x, y, value) {
    const cells = state.map.width * state.map.height;
    values[channel * cells + cellIndex(state, x, y)] = value;
  }

  function encodeStateVector(state) {
    const cells = state.map.width * state.map.height;
    const values = new Array(cells * 13).fill(0);

    for (const key of state.map.blockers) {
      const [x, y] = parseCoord(key);
      setEncodedCell(values, state, 0, x, y, 1);
    }
    for (const key of state.map.blueDeploy) {
      const [x, y] = parseCoord(key);
      setEncodedCell(values, state, 1, x, y, 1);
    }
    for (const key of state.map.redDeploy) {
      const [x, y] = parseCoord(key);
      setEncodedCell(values, state, 2, x, y, 1);
    }

    for (const unit of state.units) {
      const offset = unit.side === Side.BLUE ? 3 : 8;
      setEncodedCell(values, state, offset, unit.x, unit.y, unitKindValue(unit));
      setEncodedCell(values, state, offset + 1, unit.x, unit.y, Math.max(0, Math.min(1, unit.hp / Math.max(unitMaxHp(unit), 1))));
      setEncodedCell(values, state, offset + 2, unit.x, unit.y, Math.max(0, Math.min(1, unitDamage(unit) / 100)));
      setEncodedCell(values, state, offset + 3, unit.x, unit.y, unit.attackedThisTurn ? 1 : 0);
      setEncodedCell(values, state, offset + 4, unit.x, unit.y, isCommander(unit) ? 1 : 0);
    }

    const maxHalfTurns = Math.max(CONFIG.maxHalfTurns, 1);
    const maxActions = Math.max(CONFIG.maxActions, 1);
    const maxHand = Math.max(CONFIG.maxHandSize, 1);
    const maxUnitDeck = Math.max(state.unitDecks[Side.BLUE].length + state.hands[Side.BLUE].length, 1);
    const maxItemDeck = Math.max(state.itemDecks[Side.BLUE].length + state.hands[Side.BLUE].length, 1);
    return values.concat([
      state.currentSide === Side.BLUE ? 1 : -1,
      state.actionsLeft / maxActions,
      Math.min(state.halfTurnsPlayed / maxHalfTurns, 1),
      state.hands[Side.BLUE].length / maxHand,
      state.hands[Side.RED].length / maxHand,
      state.unitDecks[Side.BLUE].length / maxUnitDeck,
      state.unitDecks[Side.RED].length / maxUnitDeck,
      state.itemDecks[Side.BLUE].length / maxItemDeck,
      state.itemDecks[Side.RED].length / maxItemDeck,
      commanderHp(state, Side.BLUE) / Math.max(commanderMaxHp(state, Side.BLUE), 1),
      commanderHp(state, Side.RED) / Math.max(commanderMaxHp(state, Side.RED), 1),
      forwardPressure(state, Side.BLUE) / Math.max(state.map.width * 5, 1),
      forwardPressure(state, Side.RED) / Math.max(state.map.width * 5, 1),
      unitsForSide(state, Side.BLUE).length / 8,
      unitsForSide(state, Side.RED).length / 8,
    ]);
  }

  function predictValue(model, state) {
    const inputs = encodeStateVector(state);
    let output = model.b2 || 0;
    for (let hiddenIndex = 0; hiddenIndex < model.hidden_size; hiddenIndex += 1) {
      const row = model.w1[hiddenIndex];
      let activation = model.b1[hiddenIndex] || 0;
      for (let index = 0; index < inputs.length; index += 1) {
        if (inputs[index]) {
          activation += row[index] * inputs[index];
        }
      }
      output += model.w2[hiddenIndex] * Math.tanh(activation);
    }
    return Math.tanh(output);
  }

  function predictStateForPlayer(model, state, player) {
    if (state.winner) {
      return state.winner === player ? 1 : -1;
    }
    const prediction = predictValue(model, state);
    return state.currentSide === player ? prediction : -prediction;
  }

  function scoreNeuralAction(before, after, action, player) {
    const heuristicScore = scoreAction(before, after, action, player);
    const neuralScore = predictStateForPlayer(STRONGEST_VALUE_MODEL, after, player) * CONFIG.neuralScale;
    return heuristicScore * CONFIG.heuristicScale + neuralScore;
  }

  function chooseNeuralGreedyAction(state) {
    if (!STRONGEST_VALUE_MODEL) {
      return chooseHeuristicAiAction(state);
    }
    const currentLegal = legalActions(state);
    const player = state.currentSide;
    const scored = currentLegal.map((action, index) => {
      const simulated = cloneState(state);
      const legal = legalActions(simulated).find((candidate) => actionsEqual(candidate, action));
      if (!legal) {
        return { score: Number.NEGATIVE_INFINITY, tie: -index, action };
      }
      applyAction(simulated, legal);
      return { score: scoreNeuralAction(state, simulated, action, player), tie: -index, action };
    });
    scored.sort((a, b) => b.score - a.score || b.tie - a.tie);
    return scored[0]?.action || currentLegal[0];
  }

  function chooseNeuralAiAction(state) {
    if (!STRONGEST_VALUE_MODEL) {
      return chooseHeuristicAiAction(state);
    }
    const currentLegal = legalActions(state);
    const player = state.currentSide;
    const decay = 0.92;
    let frontier = [];
    const paths = [];

    currentLegal.forEach((action, index) => {
      const simulated = cloneState(state);
      applyAction(simulated, action);
      const path = {
        score: scoreNeuralAction(state, simulated, action, player),
        tie: -index,
        firstAction: action,
        state: simulated,
      };
      frontier.push(path);
      paths.push(path);
    });

    frontier.sort((a, b) => b.score - a.score || b.tie - a.tie);
    frontier = frontier.slice(0, CONFIG.neuralSearchWidth);

    for (let depth = 1; depth < CONFIG.neuralSearchDepth; depth += 1) {
      const expanded = [];
      for (const path of frontier) {
        if (path.state.winner || path.state.currentSide !== player) {
          continue;
        }
        for (const action of legalActions(path.state)) {
          const simulated = cloneState(path.state);
          applyAction(simulated, action);
          const nextPath = {
            score: path.score + scoreNeuralAction(path.state, simulated, action, player) * decay ** depth,
            tie: path.tie,
            firstAction: path.firstAction,
            state: simulated,
          };
          expanded.push(nextPath);
          paths.push(nextPath);
        }
      }
      if (!expanded.length) {
        break;
      }
      expanded.sort((a, b) => b.score - a.score || b.tie - a.tie);
      frontier = expanded.slice(0, CONFIG.neuralSearchWidth);
    }

    paths.sort((a, b) => b.score - a.score || b.tie - a.tie);
    return paths[0]?.firstAction || chooseNeuralGreedyAction(state);
  }

  function chooseHeuristicAiAction(state) {
    const currentLegal = legalActions(state);
    const player = state.currentSide;
    const decay = 0.92;
    let frontier = [];
    const paths = [];

    currentLegal.forEach((action, index) => {
      const simulated = cloneState(state);
      const legal = legalActions(simulated).find((candidate) => actionsEqual(candidate, action));
      if (!legal) {
        return;
      }
      applyAction(simulated, legal);
      const score = scoreAction(state, simulated, action, player);
      const path = { score, tie: -index, firstAction: action, state: simulated };
      frontier.push(path);
      paths.push(path);
    });

    frontier.sort((a, b) => b.score - a.score || b.tie - a.tie);
    frontier = frontier.slice(0, CONFIG.aiSearchWidth);

    for (let depth = 1; depth < CONFIG.aiSearchDepth; depth += 1) {
      const expanded = [];
      for (const path of frontier) {
        if (path.state.winner || path.state.currentSide !== player) {
          continue;
        }
        for (const action of legalActions(path.state)) {
          const simulated = cloneState(path.state);
          applyAction(simulated, action);
          const stepScore = scoreAction(path.state, simulated, action, player);
          const nextPath = {
            score: path.score + stepScore * decay ** depth,
            tie: path.tie,
            firstAction: path.firstAction,
            state: simulated,
          };
          expanded.push(nextPath);
          paths.push(nextPath);
        }
      }
      if (!expanded.length) {
        break;
      }
      expanded.sort((a, b) => b.score - a.score || b.tie - a.tie);
      frontier = expanded.slice(0, CONFIG.aiSearchWidth);
    }

    paths.sort((a, b) => b.score - a.score || b.tie - a.tie);
    return paths[0]?.firstAction || currentLegal[0];
  }

  function chooseAiAction(state) {
    if (state.currentSide === Side.RED) {
      return chooseNeuralAiAction(state);
    }
    return chooseHeuristicAiAction(state);
  }

  function spriteSheetFor(side, blueprintKey) {
    return SPRITE_SHEETS[side]?.[blueprintKey] || SPRITE_SHEETS.blue.commander;
  }

  function spriteWindowStyle(side, blueprintKey) {
    const sheet = spriteSheetFor(side, blueprintKey);
    return `--frame-ratio:${sheet.frameRatio};`;
  }

  function spriteStripStyle(frameIndex) {
    return `--frame-index:${frameIndex};`;
  }

  const dom = {
    board: document.getElementById("board"),
    hand: document.getElementById("hand"),
    log: document.getElementById("log"),
    drawUnitBtn: document.getElementById("draw-unit-btn"),
    drawItemBtn: document.getElementById("draw-item-btn"),
    endTurnBtn: document.getElementById("end-turn-btn"),
    resetBtn: document.getElementById("reset-btn"),
    turnLabel: document.getElementById("turn-label"),
    actionsLabel: document.getElementById("actions-label"),
    blueCommanderHp: document.getElementById("blue-commander-hp"),
    redCommanderHp: document.getElementById("red-commander-hp"),
    blueDeckCounts: document.getElementById("blue-deck-counts"),
    redDeckCounts: document.getElementById("red-deck-counts"),
    turnNumber: document.getElementById("turn-number"),
    logCount: document.getElementById("log-count"),
    handCount: document.getElementById("hand-count"),
    handTitle: document.getElementById("hand-title"),
    selectionCard: document.getElementById("selection-card"),
    selectionKind: document.getElementById("selection-kind"),
    winnerLabel: document.getElementById("winner-label"),
    mapLabel: document.getElementById("map-label"),
    turnCard: document.getElementById("turn-card"),
    watchAiBtn: document.getElementById("watch-ai-btn"),
    playBlueBtn: document.getElementById("play-blue-btn"),
    pauseBtn: document.getElementById("pause-btn"),
    speedBtn: document.getElementById("speed-btn"),
    modeLabel: document.getElementById("mode-label"),
    matchModeLabel: document.getElementById("match-mode-label"),
    aiSearchLabel: document.getElementById("ai-search-label"),
  };

  const app = {
    state: createInitialState(),
    dragContext: null,
    hoveredAction: null,
    pendingAi: null,
    unitAnimations: new Map(),
    itemEffects: [],
    animationTimer: null,
    mode: "play",
    paused: false,
    speedIndex: 0,
  };

  function isAiControlled(side) {
    return app.mode === "watch" || side === Side.RED;
  }

  function currentAiDelay() {
    return CONFIG.aiDelays[app.speedIndex] || CONFIG.aiDelays[0];
  }

  function setMode(mode) {
    app.mode = mode;
    app.paused = false;
    if (app.pendingAi) {
      clearTimeout(app.pendingAi);
      app.pendingAi = null;
    }
    clearDragState();
    render();
  }

  function setPaused(paused) {
    app.paused = paused;
    if (app.pendingAi) {
      clearTimeout(app.pendingAi);
      app.pendingAi = null;
    }
    render();
  }

  function setSelectionContent(title, body, meta = []) {
    dom.selectionCard.classList.remove("empty");
    dom.selectionCard.innerHTML = `
      <h3>${title}</h3>
      <p>${body}</p>
      <div class="selection-meta">
        ${meta.map((item) => `<div><strong>${item.value}</strong><span>${item.label}</span></div>`).join("")}
      </div>
    `;
  }

  function resetSelectionContent() {
    dom.selectionKind.textContent = "Nothing selected";
    dom.selectionCard.classList.add("empty");
    dom.selectionCard.innerHTML = "<p>Select or start dragging a card or unit to preview its valid targets.</p>";
  }

  function describeDragContext(context) {
    if (!context) {
      resetSelectionContent();
      return;
    }
    if (context.type === "card") {
      const card = app.state.hands[Side.BLUE][context.handIndex];
      if (!card) {
        resetSelectionContent();
        return;
      }
      if (card.kind === CardKind.UNIT) {
        const unit = UNIT_BLUEPRINTS[card.key];
        dom.selectionKind.textContent = "Dragging Unit Card";
        setSelectionContent(
          unit.name,
          "Drop onto a highlighted blue deploy square to summon this unit.",
          [
            { label: "Cost", value: `${unit.deployCost} AP` },
            { label: "Move", value: `${unit.moveRange}` },
            { label: "Range", value: `${unit.attackRange}` },
            { label: "Damage", value: `${unit.damage}` },
          ]
        );
      } else {
        const item = ITEM_BLUEPRINTS[card.key];
        dom.selectionKind.textContent = "Dragging Item Card";
        setSelectionContent(
          item.name,
          item.description,
          [
            { label: "Cost", value: `${item.cost} AP` },
            { label: "Power", value: `${item.power}` },
          ]
        );
      }
    } else if (context.type === "unit") {
      const unit = app.state.units.find((entry) => entry.unitId === context.unitId);
      if (!unit) {
        resetSelectionContent();
        return;
      }
      dom.selectionKind.textContent = "Dragging Unit";
      setSelectionContent(
        unitName(unit),
        unitRole(unit) === AttackRole.HEALER
          ? "Drop onto a highlighted square to move, or onto a glowing ally to heal."
          : "Drop onto a highlighted square to move, or onto a glowing enemy to attack.",
        [
          { label: "HP", value: `${unit.hp}/${unitMaxHp(unit)}` },
          { label: "Damage", value: `${unitDamage(unit)}` },
          { label: "Move", value: `${blueprintForUnit(unit).moveRange}` },
          { label: "Range", value: `${blueprintForUnit(unit).attackRange}` },
        ]
      );
    }
  }

  function dragTargetsForContext(context) {
    const allActions = legalActions(app.state);
    if (!context || app.state.currentSide !== Side.BLUE || app.state.winner || isAiControlled(Side.BLUE)) {
      return [];
    }
    if (context.type === "card") {
      return allActions.filter((action) => {
        if (context.handIndex == null) {
          return false;
        }
        return action.handIndex === context.handIndex && (action.kind === "deploy" || action.kind === "play_item");
      });
    }
    if (context.type === "unit") {
      return allActions.filter((action) => action.unitId === context.unitId && (action.kind === "move" || action.kind === "attack"));
    }
    return [];
  }

  function highlightMap(actions) {
    const cells = dom.board.querySelectorAll(".cell");
    cells.forEach((cell) => {
      cell.classList.remove("highlight-deploy", "highlight-move", "highlight-attack", "highlight-heal", "highlight-item");
    });

    actions.forEach((action) => {
      let targetKey = null;
      let targetClass = null;
      if (action.kind === "deploy") {
        targetKey = coordKey(action.destination[0], action.destination[1]);
        targetClass = "highlight-deploy";
      } else if (action.kind === "move") {
        targetKey = coordKey(action.destination[0], action.destination[1]);
        targetClass = "highlight-move";
      } else if (action.kind === "attack") {
        const target = app.state.units.find((unit) => unit.unitId === action.targetUnitId);
        if (target) {
          targetKey = coordKey(target.x, target.y);
          const actor = app.state.units.find((unit) => unit.unitId === action.unitId);
          targetClass = actor && unitRole(actor) === AttackRole.HEALER ? "highlight-heal" : "highlight-attack";
        }
      } else if (action.kind === "play_item") {
        const target = app.state.units.find((unit) => unit.unitId === action.targetUnitId);
        if (target) {
          targetKey = coordKey(target.x, target.y);
          targetClass = "highlight-item";
        }
      }
      if (targetKey && targetClass) {
        const cell = dom.board.querySelector(`[data-key="${targetKey}"]`);
        if (cell) {
          cell.classList.add(targetClass);
        }
      }
    });
  }

  function findDroppedAction(x, y) {
    const actions = dragTargetsForContext(app.dragContext);
    return actions.find((action) => {
      if (action.destination) {
        return action.destination[0] === x && action.destination[1] === y;
      }
      const target = app.state.units.find((unit) => unit.unitId === action.targetUnitId);
      return target && target.x === x && target.y === y;
    }) || null;
  }

  function clearDragState() {
    app.dragContext = null;
    app.hoveredAction = null;
    highlightMap([]);
    resetSelectionContent();
  }

  function pruneUnitAnimations() {
    const now = performance.now();
    for (const [unitId, animation] of app.unitAnimations.entries()) {
      if (now >= animation.startedAt + animation.durationMs) {
        app.unitAnimations.delete(unitId);
      }
    }
  }

  function pruneItemEffects() {
    const now = performance.now();
    app.itemEffects = app.itemEffects.filter((effect) => now < effect.startedAt + effect.durationMs);
  }

  function ensureAnimationLoop() {
    if (app.animationTimer) {
      return;
    }

    const tick = () => {
      app.animationTimer = null;
      pruneUnitAnimations();
      pruneItemEffects();
      render();
      if (app.unitAnimations.size > 0 || app.itemEffects.length > 0) {
        app.animationTimer = window.setTimeout(tick, 90);
      }
    };

    app.animationTimer = window.setTimeout(tick, 90);
  }

  function startUnitAnimation(unitId, kind = "attack") {
    app.unitAnimations.set(unitId, {
      kind,
      startedAt: performance.now(),
      durationMs: kind === "heal" ? 700 : 560,
      sequence: kind === "heal" ? [0, 1, 2, 3, 4, 3, 0] : [0, 1, 2, 3, 4, 0],
    });
    ensureAnimationLoop();
  }

  function startItemEffectAt(itemKey, x, y) {
    const asset = ITEM_ASSETS[itemKey];
    if (!asset) {
      return;
    }
    app.itemEffects.push({
      itemKey,
      x,
      y,
      startedAt: performance.now(),
      durationMs: 620,
      sequence: itemKey === "strength_tonic" ? [0, 1, 2, 3, 4, 4, 3] : [0, 1, 2, 3, 4, 4],
    });
    ensureAnimationLoop();
  }

  function frameForUnit(unit) {
    const activeAnimation = app.unitAnimations.get(unit.unitId);
    if (activeAnimation) {
      const elapsed = performance.now() - activeAnimation.startedAt;
      const progress = Math.min(0.999, elapsed / activeAnimation.durationMs);
      const frameIndex = Math.floor(progress * activeAnimation.sequence.length);
      return activeAnimation.sequence[frameIndex];
    }
    return 0;
  }

  function frameForCard(card, handIndex) {
    if (card.kind !== CardKind.UNIT) {
      return 0;
    }
    return 0;
  }

  function itemEffectAssetFor(itemKey) {
    return ITEM_ASSETS[itemKey] || ITEM_ASSETS.fireball;
  }

  function frameForItemEffect(effect) {
    const elapsed = performance.now() - effect.startedAt;
    const progress = Math.min(0.999, elapsed / effect.durationMs);
    const frameIndex = Math.floor(progress * effect.sequence.length);
    return effect.sequence[frameIndex];
  }

  function maybeScheduleAiTurn() {
    if (app.state.winner || app.paused || !isAiControlled(app.state.currentSide)) {
      return;
    }
    if (app.pendingAi) {
      return;
    }
    app.pendingAi = setTimeout(() => {
      runAiTurn();
    }, currentAiDelay());
  }

  function runAiTurn() {
    app.pendingAi = null;
    if (app.state.winner || app.paused || !isAiControlled(app.state.currentSide)) {
      return;
    }
    const nextAction = chooseAiAction(app.state);
    performAction(nextAction);
  }

  function performAction(action) {
    if (!action) {
      return;
    }
    if (action.kind === "attack") {
      const actor = app.state.units.find((unit) => unit.unitId === action.unitId);
      if (actor) {
        startUnitAnimation(actor.unitId, unitRole(actor) === AttackRole.HEALER ? "heal" : "attack");
      }
    } else if (action.kind === "play_item") {
      const target = app.state.units.find((unit) => unit.unitId === action.targetUnitId);
      const card = app.state.hands[app.state.currentSide][action.handIndex];
      if (target && card) {
        startItemEffectAt(card.key, target.x, target.y);
      }
    }
    applyAction(app.state, action);
    render();
    maybeScheduleAiTurn();
  }

  function renderBoard() {
    dom.board.innerHTML = "";
    const map = app.state.map;
    for (let y = 0; y < map.height; y += 1) {
      for (let x = 0; x < map.width; x += 1) {
        const cell = document.createElement("div");
        cell.className = "cell";
        cell.dataset.key = coordKey(x, y);
        cell.dataset.coord = `${x},${y}`;

        if (map.blockers.has(coordKey(x, y))) {
          cell.classList.add("blocker");
          cell.textContent = "*";
        } else if (map.blueDeploy.has(coordKey(x, y))) {
          cell.classList.add("deploy-blue");
        } else if (map.redDeploy.has(coordKey(x, y))) {
          cell.classList.add("deploy-red");
        }

        const unit = unitAt(app.state, x, y);
        if (unit) {
          const token = document.createElement("div");
          token.className = `unit-token ${unit.side}`;
          if (unit.attackedThisTurn) {
            token.classList.add("spent");
          }
          if (app.dragContext?.type === "unit" && app.dragContext.unitId === unit.unitId) {
            token.classList.add("selected");
          }
          token.draggable = unit.side === Side.BLUE && app.state.currentSide === Side.BLUE && !app.state.winner && !isAiControlled(Side.BLUE);
          token.dataset.unitId = String(unit.unitId);
          const unitSheet = spriteSheetFor(unit.side, unit.blueprintKey);
          token.innerHTML = `
            <span class="token-side">${Side.short(unit.side)}</span>
            <span class="piece-window token-window" style="${spriteWindowStyle(unit.side, unit.blueprintKey)}" aria-hidden="true">
              <img class="piece-strip token-sprite side-${unit.side}" src="${unitSheet.src}" style="${spriteStripStyle(frameForUnit(unit))}" alt="">
            </span>
            <span class="token-hp">${unit.hp}</span>
          `;
          token.addEventListener("click", (event) => {
            event.stopPropagation();
            if (unit.side === Side.BLUE && app.state.currentSide === Side.BLUE && !app.state.winner && !isAiControlled(Side.BLUE)) {
              app.dragContext = { type: "unit", unitId: unit.unitId };
              describeDragContext(app.dragContext);
              highlightMap(dragTargetsForContext(app.dragContext));
              render();
              return;
            }
            dom.selectionKind.textContent = unit.side === Side.BLUE ? "Blue Unit" : "Red Unit";
            setSelectionContent(
              unitName(unit),
              unitRole(unit) === AttackRole.HEALER ? "Support unit with ranged healing." : "Combat unit ready for tactical play.",
              [
                { label: "HP", value: `${unit.hp}/${unitMaxHp(unit)}` },
                { label: "Damage", value: `${unitDamage(unit)}` },
                { label: "Move", value: `${blueprintForUnit(unit).moveRange}` },
                { label: "Range", value: `${blueprintForUnit(unit).attackRange}` },
              ]
            );
          });
          if (token.draggable) {
            token.addEventListener("dragstart", () => {
              app.dragContext = { type: "unit", unitId: unit.unitId };
              describeDragContext(app.dragContext);
              highlightMap(dragTargetsForContext(app.dragContext));
              token.classList.add("dragging");
            });
            token.addEventListener("dragend", () => {
              token.classList.remove("dragging");
              clearDragState();
            });
          }
          cell.appendChild(token);
        }

        const cellEffect = app.itemEffects.find((effect) => effect.x === x && effect.y === y);
        if (cellEffect) {
          const effectAsset = itemEffectAssetFor(cellEffect.itemKey);
          const effectEl = document.createElement("span");
          effectEl.className = "piece-window cell-effect-window";
          effectEl.style.cssText = `--frame-ratio:${effectAsset.effectFrameRatio};`;
          effectEl.innerHTML = `
            <img class="piece-strip cell-effect-strip" src="${effectAsset.effectSrc}" style="${spriteStripStyle(frameForItemEffect(cellEffect))}" alt="">
          `;
          cell.appendChild(effectEl);
        }

        cell.addEventListener("dragover", (event) => {
          const potentialAction = findDroppedAction(x, y);
          if (!potentialAction) {
            return;
          }
          event.preventDefault();
          app.hoveredAction = potentialAction;
        });

        cell.addEventListener("drop", (event) => {
          const potentialAction = findDroppedAction(x, y);
          if (!potentialAction) {
            return;
          }
          event.preventDefault();
          performAction(potentialAction);
          clearDragState();
        });

        cell.addEventListener("click", () => {
          const potentialAction = findDroppedAction(x, y);
          if (!potentialAction) {
            return;
          }
          performAction(potentialAction);
          clearDragState();
        });

        dom.board.appendChild(cell);
      }
    }

    if (app.dragContext) {
      highlightMap(dragTargetsForContext(app.dragContext));
    }
  }

  function renderHand() {
    dom.hand.innerHTML = "";
    const visibleSide = app.mode === "watch" ? app.state.currentSide : Side.BLUE;
    const hand = app.state.hands[visibleSide];
    dom.handTitle.textContent = `${visibleSide === Side.BLUE ? "Blue" : "Red"} Hand`;
    hand.forEach((card, handIndex) => {
      const cardEl = document.createElement("article");
      cardEl.className = `card ${card.kind}`;
      if (app.dragContext?.type === "card" && app.dragContext.handIndex === handIndex) {
        cardEl.classList.add("selected");
      }
      cardEl.draggable = visibleSide === Side.BLUE && app.state.currentSide === Side.BLUE && !app.state.winner && !isAiControlled(Side.BLUE);

      if (card.kind === CardKind.UNIT) {
        const unit = UNIT_BLUEPRINTS[card.key];
        const cardSheet = spriteSheetFor(visibleSide, card.key);
        cardEl.innerHTML = `
          <div class="card-figure">
            <span class="piece-window card-window" style="${spriteWindowStyle(visibleSide, card.key)}" aria-hidden="true">
              <img class="piece-strip card-sprite side-${visibleSide}" src="${cardSheet.src}" style="${spriteStripStyle(frameForCard(card, handIndex))}" alt="">
            </span>
          </div>
          <div class="card-title">
            <div>
              <strong>${unit.name}</strong>
              <span>Unit</span>
            </div>
            <strong>${unit.deployCost} AP</strong>
          </div>
          <p>${unit.role === AttackRole.HEALER ? "Backline support and ranged healing." : "Tactical battlefield unit."}</p>
          <div class="card-stats">
            <span class="pill">DMG ${unit.damage}</span>
            <span class="pill">MOV ${unit.moveRange}</span>
            <span class="pill">RNG ${unit.attackRange}</span>
          </div>
        `;
      } else {
        const item = ITEM_BLUEPRINTS[card.key];
        const asset = ITEM_ASSETS[card.key];
        cardEl.innerHTML = `
          <div class="card-figure item-figure">
            <img class="item-icon" src="${asset.iconSrc}" alt="">
          </div>
          <div class="card-title">
            <div>
              <strong>${item.name}</strong>
              <span>Item</span>
            </div>
            <strong>${item.cost} AP</strong>
          </div>
          <p>${item.description}</p>
          <div class="card-stats">
            <span class="pill">PWR ${item.power}</span>
          </div>
        `;
      }

      cardEl.addEventListener("click", (event) => {
        event.stopPropagation();
        if (visibleSide !== Side.BLUE || isAiControlled(Side.BLUE)) {
          return;
        }
        app.dragContext = { type: "card", handIndex };
        describeDragContext(app.dragContext);
        highlightMap(dragTargetsForContext(app.dragContext));
        render();
      });

      if (cardEl.draggable) {
        cardEl.addEventListener("dragstart", () => {
          app.dragContext = { type: "card", handIndex };
          describeDragContext(app.dragContext);
          highlightMap(dragTargetsForContext(app.dragContext));
          cardEl.classList.add("dragging");
        });
        cardEl.addEventListener("dragend", () => {
          cardEl.classList.remove("dragging");
          clearDragState();
        });
      }

      dom.hand.appendChild(cardEl);
    });
  }

  function renderLog() {
    dom.log.innerHTML = "";
    [...app.state.log].slice(-14).reverse().forEach((entry) => {
      const line = document.createElement("div");
      line.className = "log-entry";
      line.textContent = entry;
      dom.log.appendChild(line);
    });
  }

  function renderStatus() {
    const blueCommander = findCommander(app.state, Side.BLUE);
    const redCommander = findCommander(app.state, Side.RED);
    dom.turnLabel.textContent = `${app.state.currentSide === Side.BLUE ? "Blue" : "Red"} Turn`;
    dom.actionsLabel.textContent = `${app.state.actionsLeft} AP`;
    dom.blueCommanderHp.textContent = `${blueCommander ? blueCommander.hp : 0} / ${UNIT_BLUEPRINTS.commander.maxHp}`;
    dom.redCommanderHp.textContent = `${redCommander ? redCommander.hp : 0} / ${UNIT_BLUEPRINTS.commander.maxHp}`;
    dom.blueDeckCounts.textContent = `${app.state.unitDecks[Side.BLUE].length}U / ${app.state.itemDecks[Side.BLUE].length}I`;
    dom.redDeckCounts.textContent = `${app.state.unitDecks[Side.RED].length}U / ${app.state.itemDecks[Side.RED].length}I`;
    dom.turnNumber.textContent = String(app.state.turnNumber);
    dom.logCount.textContent = String(app.state.log.length);
    const visibleHandSide = app.mode === "watch" ? app.state.currentSide : Side.BLUE;
    dom.handCount.textContent = `${app.state.hands[visibleHandSide].length} cards`;
    dom.mapLabel.textContent = app.state.map.name;
    dom.aiSearchLabel.textContent = STRONGEST_VALUE_MODEL
      ? `Red neural beam ${CONFIG.neuralSearchWidth}x${CONFIG.neuralSearchDepth}`
      : `Beam ${CONFIG.aiSearchWidth}`;
    dom.modeLabel.textContent = app.mode === "watch" ? "Heuristic vs Neural" : "Human vs Neural";
    dom.matchModeLabel.textContent = app.mode === "watch" ? "Heuristic vs Neural" : "Human vs Neural";

    if (app.state.winner) {
      dom.winnerLabel.textContent = `${app.state.winner === Side.BLUE ? "Blue" : "Red"} wins`;
    } else if (app.paused) {
      dom.winnerLabel.textContent = "Paused";
    } else if (isAiControlled(app.state.currentSide)) {
      dom.winnerLabel.textContent = app.state.currentSide === Side.RED ? "Neural Thinking" : "AI Thinking";
    } else {
      dom.winnerLabel.textContent = "Match Active";
    }

    const blueLegal = legalActions(app.state);
    const humanCanAct = app.state.currentSide === Side.BLUE && !isAiControlled(Side.BLUE) && !app.state.winner;
    dom.drawUnitBtn.disabled = !blueLegal.some((action) => action.kind === "draw_unit") || !humanCanAct;
    dom.drawItemBtn.disabled = !blueLegal.some((action) => action.kind === "draw_item") || !humanCanAct;
    dom.endTurnBtn.disabled = !humanCanAct;
    dom.watchAiBtn.classList.toggle("active", app.mode === "watch");
    dom.playBlueBtn.classList.toggle("active", app.mode === "play");
    dom.pauseBtn.disabled = app.state.winner || app.mode !== "watch";
    dom.pauseBtn.textContent = app.paused ? "Resume" : "Pause";
    dom.speedBtn.textContent = `Speed ${app.speedIndex + 1}x`;
  }

  function render() {
    renderBoard();
    renderHand();
    renderLog();
    renderStatus();
    if (!app.dragContext) {
      resetSelectionContent();
    }
    maybeScheduleAiTurn();
  }

  dom.drawUnitBtn.addEventListener("click", () => {
    if (isAiControlled(Side.BLUE)) {
      return;
    }
    const action = legalActions(app.state).find((entry) => entry.kind === "draw_unit");
    if (action) {
      performAction(action);
    }
  });

  dom.drawItemBtn.addEventListener("click", () => {
    if (isAiControlled(Side.BLUE)) {
      return;
    }
    const action = legalActions(app.state).find((entry) => entry.kind === "draw_item");
    if (action) {
      performAction(action);
    }
  });

  dom.endTurnBtn.addEventListener("click", () => {
    if (isAiControlled(Side.BLUE)) {
      return;
    }
    const action = legalActions(app.state).find((entry) => entry.kind === "end_turn");
    if (action) {
      performAction(action);
    }
  });

  dom.resetBtn.addEventListener("click", () => {
    if (app.pendingAi) {
      clearTimeout(app.pendingAi);
    }
    if (app.animationTimer) {
      clearTimeout(app.animationTimer);
      app.animationTimer = null;
    }
    app.unitAnimations.clear();
    app.itemEffects = [];
    app.state = createInitialState();
    clearDragState();
    render();
  });

  dom.watchAiBtn.addEventListener("click", () => {
    setMode("watch");
  });

  dom.playBlueBtn.addEventListener("click", () => {
    setMode("play");
  });

  dom.pauseBtn.addEventListener("click", () => {
    setPaused(!app.paused);
  });

  dom.speedBtn.addEventListener("click", () => {
    app.speedIndex = (app.speedIndex + 1) % CONFIG.aiDelays.length;
    if (app.pendingAi) {
      clearTimeout(app.pendingAi);
      app.pendingAi = null;
    }
    render();
  });

  document.addEventListener("click", (event) => {
    const inCard = event.target.closest(".card");
    const inToken = event.target.closest(".unit-token");
    const inControl = event.target.closest(".action-btn");
    if (!inCard && !inToken && !inControl) {
      clearDragState();
      render();
    }
  });

  render();
})();

