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
    commander: { key: "commander", name: "Commander", glyph: "C", maxHp: 100, damage: 20, moveRange: 2, attackRange: 1, deployCost: 0, role: AttackRole.MELEE, canDeploy: false },
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
  };

  const DEFAULT_WEIGHTS = {
    bias: 0,
    enemyCommanderDelta: 14,
    ownCommanderDelta: -16,
    enemyUnitDelta: 48,
    ownUnitDelta: -52,
    enemyTotalHpDelta: 1.7,
    ownTotalHpDelta: -1.3,
    forwardPressureDelta: 2,
    handDelta: 0.8,
    deploy: 6,
    move: 1,
    attack: 4,
    heal: 3.5,
    item: 3,
    drawUnit: 1.8,
    drawItem: 1,
    endTurn: -2.5,
    remainingAp: 0.2,
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
    return {
      bias: 1,
      enemyCommanderDelta: commanderHp(before, enemy) - commanderHp(after, enemy),
      ownCommanderDelta: commanderHp(before, player) - commanderHp(after, player),
      enemyUnitDelta: unitsForSide(before, enemy).length - unitsForSide(after, enemy).length,
      ownUnitDelta: unitsForSide(before, player).length - unitsForSide(after, player).length,
      enemyTotalHpDelta: totalHp(before, enemy) - totalHp(after, enemy),
      ownTotalHpDelta: totalHp(before, player) - totalHp(after, player),
      forwardPressureDelta: forwardPressure(after, player) - forwardPressure(before, player),
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

  function chooseAiAction(state) {
    const currentLegal = legalActions(state);
    const player = state.currentSide;
    let best = currentLegal[0];
    let bestScore = -Infinity;
    currentLegal.forEach((action) => {
      const simulated = cloneState(state);
      const legal = legalActions(simulated).find((candidate) => actionsEqual(candidate, action));
      if (!legal) {
        return;
      }
      applyAction(simulated, legal);
      const score = scoreAction(state, simulated, action, player);
      if (score > bestScore) {
        bestScore = score;
        best = action;
      }
    });
    return best;
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
    selectionCard: document.getElementById("selection-card"),
    selectionKind: document.getElementById("selection-kind"),
    winnerLabel: document.getElementById("winner-label"),
    mapLabel: document.getElementById("map-label"),
    turnCard: document.getElementById("turn-card"),
  };

  const app = {
    state: createInitialState(),
    dragContext: null,
    hoveredAction: null,
    pendingAi: null,
    unitAnimations: new Map(),
    itemEffects: [],
    animationTimer: null,
  };

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
    if (!context || app.state.currentSide !== Side.BLUE || app.state.winner) {
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
    if (app.state.winner || app.state.currentSide !== Side.RED) {
      return;
    }
    if (app.pendingAi) {
      clearTimeout(app.pendingAi);
    }
    app.pendingAi = setTimeout(() => {
      runAiTurn();
    }, 420);
  }

  function runAiTurn() {
    if (app.state.winner || app.state.currentSide !== Side.RED) {
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
          token.draggable = unit.side === Side.BLUE && app.state.currentSide === Side.BLUE && !app.state.winner;
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
            if (unit.side === Side.BLUE && app.state.currentSide === Side.BLUE && !app.state.winner) {
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
    const hand = app.state.hands[Side.BLUE];
    hand.forEach((card, handIndex) => {
      const cardEl = document.createElement("article");
      cardEl.className = `card ${card.kind}`;
      if (app.dragContext?.type === "card" && app.dragContext.handIndex === handIndex) {
        cardEl.classList.add("selected");
      }
      cardEl.draggable = app.state.currentSide === Side.BLUE && !app.state.winner;

      if (card.kind === CardKind.UNIT) {
        const unit = UNIT_BLUEPRINTS[card.key];
        const cardSheet = spriteSheetFor(Side.BLUE, card.key);
        cardEl.innerHTML = `
          <div class="card-figure">
            <span class="piece-window card-window" style="${spriteWindowStyle(Side.BLUE, card.key)}" aria-hidden="true">
              <img class="piece-strip card-sprite side-blue" src="${cardSheet.src}" style="${spriteStripStyle(frameForCard(card, handIndex))}" alt="">
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
    dom.handCount.textContent = `${app.state.hands[Side.BLUE].length} cards`;
    dom.mapLabel.textContent = app.state.map.name;

    if (app.state.winner) {
      dom.winnerLabel.textContent = `${app.state.winner === Side.BLUE ? "Blue" : "Red"} wins`;
    } else if (app.state.currentSide === Side.RED) {
      dom.winnerLabel.textContent = "AI Thinking";
    } else {
      dom.winnerLabel.textContent = "Match Active";
    }

    const blueLegal = legalActions(app.state);
    dom.drawUnitBtn.disabled = !blueLegal.some((action) => action.kind === "draw_unit") || app.state.currentSide !== Side.BLUE || !!app.state.winner;
    dom.drawItemBtn.disabled = !blueLegal.some((action) => action.kind === "draw_item") || app.state.currentSide !== Side.BLUE || !!app.state.winner;
    dom.endTurnBtn.disabled = app.state.currentSide !== Side.BLUE || !!app.state.winner;
  }

  function render() {
    renderBoard();
    renderHand();
    renderLog();
    renderStatus();
    if (!app.dragContext) {
      resetSelectionContent();
    }
  }

  dom.drawUnitBtn.addEventListener("click", () => {
    const action = legalActions(app.state).find((entry) => entry.kind === "draw_unit");
    if (action) {
      performAction(action);
    }
  });

  dom.drawItemBtn.addEventListener("click", () => {
    const action = legalActions(app.state).find((entry) => entry.kind === "draw_item");
    if (action) {
      performAction(action);
    }
  });

  dom.endTurnBtn.addEventListener("click", () => {
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

