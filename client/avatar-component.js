/**
 * BridgeCast AI — 3D Sign Language Avatar Component
 *
 * Embeddable JavaScript class that creates and animates a cute chibi-style
 * 3D character using Three.js to perform sign language gestures.
 *
 * Usage:
 *   const avatar = new SignLanguageAvatar(containerElement);
 *   avatar.playSign('HELLO');
 *   avatar.playSequence(['HELLO', 'NICE', 'MEET', 'YOU']);
 *   avatar.setExpression('smile');
 *   avatar.reset();
 *   avatar.dispose();
 *
 * Requires Three.js r128+ loaded before this script.
 */

(function (root, factory) {
  if (typeof define === 'function' && define.amd) {
    define([], factory);
  } else if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.SignLanguageAvatar = factory();
  }
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* ================================================================
   *  Color Palette — soft pastel BridgeCast theme
   * ================================================================ */
  /* ================================================================
   *  Avatar Themes — selectable character styles
   * ================================================================ */
  const AVATAR_THEMES = {
    pastel: {
      name: 'Pastel',
      description: 'Soft & friendly',
      skin: 0xffe0d0, skinLight: 0xfff0e8,
      hair: 0x7c6faa, hairDark: 0x5b4f8a,
      bodyMain: 0x9b8fd4, bodyLight: 0xb8aee8, bodyDark: 0x7b6fc0,
      pants: 0x6e8bce, shoes: 0x5b6baa,
      eyeBlack: 0x1a1a2e, eyeWhite: 0xffffff, eyeHighlight: 0xffffff,
      mouthPink: 0xe87b8a, blush: 0xffb0b0,
      ground: 0x1a1a2e, groundGrid: 0x2a2a4e,
    },
    highContrast: {
      name: 'High Contrast',
      description: 'Best for sign visibility',
      skin: 0xf5c6a0, skinLight: 0xfadbc4,
      hair: 0x2d2d2d, hairDark: 0x1a1a1a,
      bodyMain: 0x1a1a2e, bodyLight: 0x2a2a4e, bodyDark: 0x0e0e1f,
      pants: 0x333355, shoes: 0x222244,
      eyeBlack: 0x000000, eyeWhite: 0xffffff, eyeHighlight: 0xffffff,
      mouthPink: 0xd4626e, blush: 0xf0a0a0,
      ground: 0x0a0a14, groundGrid: 0x1a1a2e,
    },
    warm: {
      name: 'Warm Natural',
      description: 'Realistic & calm',
      skin: 0xe8c4a0, skinLight: 0xf0d4b8,
      hair: 0x4a3728, hairDark: 0x332618,
      bodyMain: 0xc4785a, bodyLight: 0xd89878, bodyDark: 0xa8604a,
      pants: 0x5a7868, shoes: 0x3a5048,
      eyeBlack: 0x1a1410, eyeWhite: 0xfff8f0, eyeHighlight: 0xffffff,
      mouthPink: 0xc46060, blush: 0xe0a098,
      ground: 0x1a1a18, groundGrid: 0x2a2a28,
    },
    ocean: {
      name: 'Ocean',
      description: 'Cool & refreshing',
      skin: 0xf0d8c8, skinLight: 0xf8e8e0,
      hair: 0x2878a8, hairDark: 0x1a5878,
      bodyMain: 0x3898c8, bodyLight: 0x58b8e0, bodyDark: 0x2878a8,
      pants: 0x2060a0, shoes: 0x184878,
      eyeBlack: 0x0a1828, eyeWhite: 0xffffff, eyeHighlight: 0xffffff,
      mouthPink: 0xd07080, blush: 0xf0b8b8,
      ground: 0x0a1420, groundGrid: 0x1a2838,
    },
    mint: {
      name: 'Mint Fresh',
      description: 'Lively & accessible',
      skin: 0xffd8c8, skinLight: 0xffe8e0,
      hair: 0x208868, hairDark: 0x186850,
      bodyMain: 0x30a878, bodyLight: 0x50c898, bodyDark: 0x208860,
      pants: 0x607888, shoes: 0x405868,
      eyeBlack: 0x102018, eyeWhite: 0xffffff, eyeHighlight: 0xffffff,
      mouthPink: 0xe07878, blush: 0xf8b0a8,
      ground: 0x101a18, groundGrid: 0x202a28,
    },
  };

  let COLORS = { ...AVATAR_THEMES.pastel };

  /* ================================================================
   *  Arm/Hand Pose Definitions — 3D positions {x, y, z}
   *  All coordinates relative to character center
   *  Right side = positive X, Up = positive Y, Forward = positive Z
   * ================================================================ */
  const ARM_POSES = {
    rest: {
      rightShoulder: { x: 0, y: 0, z: 0 },
      rightElbow:    { x: 0.35, y: -0.5, z: 0.05 },
      rightHand:     { x: 0.4, y: -1.0, z: 0.1 },
      leftShoulder:  { x: 0, y: 0, z: 0 },
      leftElbow:     { x: -0.35, y: -0.5, z: 0.05 },
      leftHand:      { x: -0.4, y: -1.0, z: 0.1 },
      rightFingers:  'relaxed',
      leftFingers:   'relaxed',
    },
    rightHigh: {
      rightElbow: { x: 0.4, y: 0.3, z: 0.2 },
      rightHand:  { x: 0.45, y: 0.8, z: 0.3 },
      rightFingers: 'open',
    },
    rightFace: {
      rightElbow: { x: 0.35, y: 0.1, z: 0.3 },
      rightHand:  { x: 0.3, y: 0.5, z: 0.4 },
      rightFingers: 'flat',
    },
    rightChin: {
      rightElbow: { x: 0.25, y: 0.0, z: 0.3 },
      rightHand:  { x: 0.15, y: 0.35, z: 0.45 },
      rightFingers: 'flat',
    },
    rightChest: {
      rightElbow: { x: 0.3, y: -0.15, z: 0.25 },
      rightHand:  { x: 0.15, y: -0.1, z: 0.4 },
      rightFingers: 'flat',
    },
    rightForward: {
      rightElbow: { x: 0.35, y: -0.1, z: 0.3 },
      rightHand:  { x: 0.5, y: 0.0, z: 0.7 },
      rightFingers: 'point',
    },
    rightForehead: {
      rightElbow: { x: 0.3, y: 0.2, z: 0.25 },
      rightHand:  { x: 0.2, y: 0.65, z: 0.35 },
      rightFingers: 'flat',
    },
    rightWave: {
      rightElbow: { x: 0.45, y: 0.35, z: 0.15 },
      rightHand:  { x: 0.55, y: 0.85, z: 0.25 },
      rightFingers: 'open',
    },
    rightMouth: {
      rightElbow: { x: 0.2, y: 0.0, z: 0.3 },
      rightHand:  { x: 0.1, y: 0.25, z: 0.5 },
      rightFingers: 'pinch',
    },
    pointSelf: {
      rightElbow: { x: 0.2, y: -0.1, z: 0.3 },
      rightHand:  { x: 0.05, y: -0.05, z: 0.45 },
      rightFingers: 'point',
    },
    leftHigh: {
      leftElbow: { x: -0.4, y: 0.3, z: 0.2 },
      leftHand:  { x: -0.45, y: 0.8, z: 0.3 },
      leftFingers: 'open',
    },
    leftChest: {
      leftElbow: { x: -0.3, y: -0.15, z: 0.25 },
      leftHand:  { x: -0.15, y: -0.1, z: 0.4 },
      leftFingers: 'flat',
    },
    leftForward: {
      leftElbow: { x: -0.35, y: -0.1, z: 0.3 },
      leftHand:  { x: -0.5, y: 0.0, z: 0.7 },
      leftFingers: 'point',
    },
    leftMid: {
      leftElbow: { x: -0.35, y: -0.3, z: 0.15 },
      leftHand:  { x: -0.4, y: -0.6, z: 0.2 },
      leftFingers: 'relaxed',
    },
    bothMidOut: {
      rightElbow: { x: 0.45, y: -0.2, z: 0.15 },
      rightHand:  { x: 0.6, y: -0.4, z: 0.2 },
      leftElbow:  { x: -0.45, y: -0.2, z: 0.15 },
      leftHand:   { x: -0.6, y: -0.4, z: 0.2 },
      rightFingers: 'relaxed',
      leftFingers: 'relaxed',
    },
    bothCenter: {
      rightElbow: { x: 0.25, y: -0.1, z: 0.3 },
      rightHand:  { x: 0.1, y: -0.05, z: 0.45 },
      leftElbow:  { x: -0.25, y: -0.1, z: 0.3 },
      leftHand:   { x: -0.1, y: -0.05, z: 0.45 },
      rightFingers: 'flat',
      leftFingers: 'flat',
    },
    bothUp: {
      rightElbow: { x: 0.35, y: 0.25, z: 0.2 },
      rightHand:  { x: 0.35, y: 0.75, z: 0.25 },
      leftElbow:  { x: -0.35, y: 0.25, z: 0.2 },
      leftHand:   { x: -0.35, y: 0.75, z: 0.25 },
      rightFingers: 'open',
      leftFingers: 'open',
    },
    bothDown: {
      rightElbow: { x: 0.3, y: -0.4, z: 0.1 },
      rightHand:  { x: 0.35, y: -0.85, z: 0.15 },
      leftElbow:  { x: -0.3, y: -0.4, z: 0.1 },
      leftHand:   { x: -0.35, y: -0.85, z: 0.15 },
      rightFingers: 'flat',
      leftFingers: 'flat',
    },
    bothSpread: {
      rightElbow: { x: 0.55, y: -0.05, z: 0.15 },
      rightHand:  { x: 0.85, y: 0.0, z: 0.2 },
      leftElbow:  { x: -0.55, y: -0.05, z: 0.15 },
      leftHand:   { x: -0.85, y: 0.0, z: 0.2 },
      rightFingers: 'open',
      leftFingers: 'open',
    },
    ilyPose: {
      rightElbow: { x: 0.45, y: 0.2, z: 0.25 },
      rightHand:  { x: 0.6, y: 0.6, z: 0.35 },
      rightFingers: 'ily',
    },
    comeTogether: {
      rightElbow: { x: 0.2, y: -0.05, z: 0.35 },
      rightHand:  { x: 0.05, y: -0.0, z: 0.5 },
      leftElbow:  { x: -0.2, y: -0.05, z: 0.35 },
      leftHand:   { x: -0.05, y: -0.0, z: 0.5 },
      rightFingers: 'fist',
      leftFingers: 'fist',
    },
  };

  /* ================================================================
   *  Sign-to-Pose Map (same glossary as 2D version)
   * ================================================================ */
  const SIGN_POSE_MAP = {
    'HELLO':      { poses: ['rightWave', 'rightHigh', 'rightWave'], expression: 'smile', headMove: 'none' },
    'THANK-YOU':  { poses: ['rightChin', 'rightForward'], expression: 'smile', headMove: 'nod' },
    'YES':        { poses: ['rightChest', 'rightChest'], expression: 'smile', headMove: 'nod' },
    'NO':         { poses: ['rightFace', 'rightFace'], expression: 'frown', headMove: 'shake' },
    'HELP':       { poses: ['bothCenter', 'bothUp'], expression: 'concerned', headMove: 'none' },
    'PLEASE':     { poses: ['rightChest', 'rightChest'], expression: 'polite', headMove: 'tilt' },
    'SORRY':      { poses: ['rightChest', 'rightChest'], expression: 'sad', headMove: 'tilt' },
    'GOOD':       { poses: ['rightChin', 'bothCenter'], expression: 'smile', headMove: 'nod' },
    'BAD':        { poses: ['rightChin', 'bothDown'], expression: 'frown', headMove: 'shake' },
    'QUESTION':   { poses: ['rightForward', 'rightForward'], expression: 'brow_raise', headMove: 'tilt' },
    'AGREE':      { poses: ['rightForehead', 'bothCenter'], expression: 'smile', headMove: 'nod' },
    'DISAGREE':   { poses: ['bothCenter', 'bothSpread'], expression: 'frown', headMove: 'shake' },
    'UNDERSTAND': { poses: ['rightForehead', 'rightHigh'], expression: 'smile', headMove: 'nod' },
    'NAME':       { poses: ['bothCenter', 'bothCenter'], expression: 'neutral', headMove: 'none' },
    'NICE':       { poses: ['bothCenter', 'rightForward'], expression: 'smile', headMove: 'none' },
    'MEET':       { poses: ['bothSpread', 'comeTogether'], expression: 'smile', headMove: 'none' },
    'GOODBYE':    { poses: ['rightWave', 'rightHigh', 'rightWave'], expression: 'smile', headMove: 'nod' },
    'WAIT':       { poses: ['bothUp', 'bothUp'], expression: 'neutral', headMove: 'none' },
    'REPEAT':     { poses: ['rightForward', 'bothCenter'], expression: 'neutral', headMove: 'none' },
    'SLOW-DOWN':  { poses: ['bothCenter', 'bothDown'], expression: 'neutral', headMove: 'none' },
    'I-LOVE-YOU': { poses: ['ilyPose', 'ilyPose'], expression: 'big_smile', headMove: 'tilt' },
    'GO':         { poses: ['bothCenter', 'bothSpread'], expression: 'neutral', headMove: 'none' },
    'COME':       { poses: ['rightForward', 'rightChest'], expression: 'smile', headMove: 'nod' },
    'WANT':       { poses: ['bothSpread', 'bothCenter'], expression: 'neutral', headMove: 'none' },
    'NEED':       { poses: ['rightForward', 'rightChest'], expression: 'neutral', headMove: 'nod' },
    'LIKE':       { poses: ['rightChest', 'rightForward'], expression: 'smile', headMove: 'none' },
    'WORK':       { poses: ['bothCenter', 'bothCenter'], expression: 'neutral', headMove: 'none' },
    'HOME':       { poses: ['rightMouth', 'rightFace'], expression: 'smile', headMove: 'tilt' },
    'SCHOOL':     { poses: ['bothSpread', 'comeTogether'], expression: 'neutral', headMove: 'none' },
    'FRIEND':     { poses: ['bothCenter', 'bothCenter'], expression: 'smile', headMove: 'none' },
    'FAMILY':     { poses: ['bothMidOut', 'bothCenter'], expression: 'smile', headMove: 'none' },
    'I':          { poses: ['pointSelf'], expression: 'neutral', headMove: 'none' },
    'YOU':        { poses: ['rightForward'], expression: 'neutral', headMove: 'none' },
    'WE':         { poses: ['pointSelf', 'rightForward'], expression: 'smile', headMove: 'none' },
    'MY':         { poses: ['rightChest'], expression: 'neutral', headMove: 'none' },
    'YOUR':       { poses: ['rightForward'], expression: 'neutral', headMove: 'none' },
    'WHAT':       { poses: ['bothSpread'], expression: 'brow_raise', headMove: 'tilt' },
    'WHERE':      { poses: ['rightForward', 'rightHigh'], expression: 'brow_raise', headMove: 'tilt' },
    'NOT':        { poses: ['rightChin', 'rightForward'], expression: 'frown', headMove: 'shake' },
    'HAPPY':      { poses: ['bothCenter', 'bothUp'], expression: 'big_smile', headMove: 'none' },
    'SAD':        { poses: ['bothUp', 'bothDown'], expression: 'sad', headMove: 'drop' },
    'KNOW':       { poses: ['rightForehead'], expression: 'neutral', headMove: 'none' },
    'THINK':      { poses: ['rightForehead'], expression: 'neutral', headMove: 'tilt' },
    'LEARN':      { poses: ['bothCenter', 'rightForehead'], expression: 'neutral', headMove: 'none' },
    'HAVE':       { poses: ['bothCenter'], expression: 'neutral', headMove: 'none' },
    'SEE':        { poses: ['rightFace', 'rightForward'], expression: 'neutral', headMove: 'none' },
    'FINISH':     { poses: ['bothCenter', 'bothSpread'], expression: 'neutral', headMove: 'none' },
    'NOW':        { poses: ['bothCenter', 'bothDown'], expression: 'neutral', headMove: 'nod' },
    'TODAY':      { poses: ['bothCenter', 'bothDown'], expression: 'neutral', headMove: 'nod' },
    'TOMORROW':   { poses: ['rightFace', 'rightForward'], expression: 'neutral', headMove: 'none' },
    'YESTERDAY':  { poses: ['rightForward', 'rightFace'], expression: 'neutral', headMove: 'none' },
    'PEOPLE':     { poses: ['bothMidOut', 'bothMidOut'], expression: 'neutral', headMove: 'none' },
    'DEAF':       { poses: ['rightFace', 'rightMouth'], expression: 'neutral', headMove: 'none' },
    'HEARING':    { poses: ['rightMouth', 'rightMouth'], expression: 'neutral', headMove: 'none' },
    'SIGN':       { poses: ['bothMidOut', 'bothMidOut'], expression: 'neutral', headMove: 'none' },
    'LANGUAGE':   { poses: ['comeTogether', 'bothSpread'], expression: 'neutral', headMove: 'none' },
    'WORLD':      { poses: ['bothMidOut', 'bothMidOut'], expression: 'neutral', headMove: 'none' },
    'MORE':       { poses: ['bothSpread', 'comeTogether'], expression: 'neutral', headMove: 'none' },
    'STOP':       { poses: ['rightHigh', 'bothCenter'], expression: 'neutral', headMove: 'none' },
    'EAT':        { poses: ['rightMouth', 'rightMouth'], expression: 'neutral', headMove: 'none' },
    'DRINK':      { poses: ['rightChin', 'rightMouth'], expression: 'neutral', headMove: 'tilt' },
    'WATER':      { poses: ['rightChin', 'rightChin'], expression: 'neutral', headMove: 'none' },
    'AGAIN':      { poses: ['rightHigh', 'bothCenter'], expression: 'neutral', headMove: 'none' },
    'TIME':       { poses: ['rightChest', 'rightChest'], expression: 'neutral', headMove: 'none' },
    'TALK':       { poses: ['rightMouth', 'rightForward'], expression: 'neutral', headMove: 'none' },
    'TEACH':      { poses: ['bothUp', 'bothSpread'], expression: 'smile', headMove: 'none' },
    'STUDENT':    { poses: ['bothCenter', 'rightForehead'], expression: 'neutral', headMove: 'none' },
    'TEACHER':    { poses: ['bothUp', 'bothSpread'], expression: 'neutral', headMove: 'none' },
    'START':      { poses: ['bothCenter', 'bothSpread'], expression: 'neutral', headMove: 'none' },
    'FOOD':       { poses: ['rightMouth', 'rightMouth'], expression: 'neutral', headMove: 'none' },
    'MONEY':      { poses: ['rightChest', 'bothCenter'], expression: 'neutral', headMove: 'none' },
    'CAN':        { poses: ['bothCenter', 'bothDown'], expression: 'smile', headMove: 'nod' },
    'WILL':       { poses: ['rightFace', 'rightForward'], expression: 'neutral', headMove: 'none' },
    'DO':         { poses: ['bothMidOut', 'bothMidOut'], expression: 'neutral', headMove: 'none' },
    'MAKE':       { poses: ['bothCenter', 'bothCenter'], expression: 'neutral', headMove: 'none' },
    'FEEL':       { poses: ['rightChest', 'rightChest'], expression: 'neutral', headMove: 'none' },

    // --- Workplace / Meeting ---
    'MEETING':    { poses: ['bothSpread', 'comeTogether'], expression: 'neutral', headMove: 'none' },
    'PROJECT':    { poses: ['bothCenter', 'bothSpread'], expression: 'neutral', headMove: 'none' },
    'TEAM':       { poses: ['bothMidOut', 'comeTogether'], expression: 'smile', headMove: 'none' },
    'BOSS':       { poses: ['rightForehead', 'rightHigh'], expression: 'neutral', headMove: 'none' },
    'EMAIL':      { poses: ['rightChest', 'rightForward'], expression: 'neutral', headMove: 'none' },
    'SCHEDULE':   { poses: ['bothCenter', 'bothDown'], expression: 'neutral', headMove: 'none' },
    'BREAK':      { poses: ['bothCenter', 'bothSpread'], expression: 'smile', headMove: 'nod' },
    'IDEA':       { poses: ['rightForehead', 'rightHigh'], expression: 'big_smile', headMove: 'none' },
    'PROBLEM':    { poses: ['rightForehead', 'rightForehead'], expression: 'concerned', headMove: 'none' },
    'SOLVE':      { poses: ['bothCenter', 'bothSpread'], expression: 'smile', headMove: 'nod' },
    'DECIDE':     { poses: ['bothUp', 'bothDown'], expression: 'neutral', headMove: 'nod' },
    'PLAN':       { poses: ['bothCenter', 'bothCenter'], expression: 'neutral', headMove: 'none' },
    'REPORT':     { poses: ['rightChest', 'rightForward'], expression: 'neutral', headMove: 'none' },
    'BUDGET':     { poses: ['rightChest', 'bothCenter'], expression: 'neutral', headMove: 'none' },
    'DEADLINE':   { poses: ['rightChest', 'rightForward'], expression: 'concerned', headMove: 'nod' },

    // --- Education ---
    'CLASS':      { poses: ['bothSpread', 'bothSpread'], expression: 'neutral', headMove: 'none' },
    'TEST':       { poses: ['bothCenter', 'bothDown'], expression: 'neutral', headMove: 'none' },
    'HOMEWORK':   { poses: ['bothCenter', 'rightChest'], expression: 'neutral', headMove: 'none' },
    'READ':       { poses: ['rightForward', 'rightDown'], expression: 'neutral', headMove: 'none' },
    'WRITE':      { poses: ['rightChest', 'rightChest'], expression: 'neutral', headMove: 'none' },
    'PRACTICE':   { poses: ['bothCenter', 'bothCenter'], expression: 'neutral', headMove: 'nod' },
    'GRADE':      { poses: ['rightChest', 'rightForward'], expression: 'neutral', headMove: 'none' },
    'BOOK':       { poses: ['bothCenter', 'bothSpread'], expression: 'neutral', headMove: 'none' },

    // --- Healthcare ---
    'DOCTOR':     { poses: ['rightChest', 'rightChest'], expression: 'neutral', headMove: 'none' },
    'HOSPITAL':   { poses: ['rightChest', 'rightChest'], expression: 'concerned', headMove: 'none' },
    'MEDICINE':   { poses: ['rightChest', 'rightMouth'], expression: 'neutral', headMove: 'none' },
    'PAIN':       { poses: ['bothCenter', 'bothCenter'], expression: 'frown', headMove: 'none' },
    'SICK':       { poses: ['rightForehead', 'rightChest'], expression: 'sad', headMove: 'none' },
    'SAFE':       { poses: ['bothCenter', 'bothSpread'], expression: 'smile', headMove: 'nod' },
    'EMERGENCY':  { poses: ['rightHigh', 'rightHigh'], expression: 'surprised', headMove: 'shake' },

    // --- Emotions / Social ---
    'ANGRY':      { poses: ['bothCenter', 'bothUp'], expression: 'frown', headMove: 'none' },
    'EXCITED':    { poses: ['bothUp', 'bothUp'], expression: 'big_smile', headMove: 'nod' },
    'TIRED':      { poses: ['bothCenter', 'bothDown'], expression: 'sad', headMove: 'drop' },
    'SURPRISED':  { poses: ['bothSpread', 'bothUp'], expression: 'surprised', headMove: 'none' },
    'PROUD':      { poses: ['rightChest', 'rightHigh'], expression: 'big_smile', headMove: 'nod' },

    // --- Accessibility ---
    'INTERPRETER':{ poses: ['bothMidOut', 'bothMidOut'], expression: 'neutral', headMove: 'none' },
    'ACCESS':     { poses: ['bothCenter', 'bothSpread'], expression: 'neutral', headMove: 'none' },
    'EQUAL':      { poses: ['bothCenter', 'bothCenter'], expression: 'smile', headMove: 'nod' },
  };

  /* ================================================================
   *  Expression Configs for 3D face
   * ================================================================ */
  const EXPRESSIONS = {
    neutral:    { eyeScaleY: 1.0, browY: 0, mouthScaleX: 0.6, mouthScaleY: 0.15, mouthY: 0 },
    smile:      { eyeScaleY: 0.8, browY: 0.02, mouthScaleX: 0.7, mouthScaleY: 0.25, mouthY: -0.01 },
    big_smile:  { eyeScaleY: 0.6, browY: 0.04, mouthScaleX: 0.8, mouthScaleY: 0.35, mouthY: -0.02 },
    frown:      { eyeScaleY: 1.0, browY: -0.03, mouthScaleX: 0.5, mouthScaleY: 0.1, mouthY: 0.03 },
    sad:        { eyeScaleY: 1.1, browY: -0.04, mouthScaleX: 0.45, mouthScaleY: 0.12, mouthY: 0.04 },
    brow_raise: { eyeScaleY: 1.2, browY: 0.06, mouthScaleX: 0.5, mouthScaleY: 0.2, mouthY: 0 },
    concerned:  { eyeScaleY: 1.05, browY: -0.02, mouthScaleX: 0.45, mouthScaleY: 0.12, mouthY: 0.02 },
    polite:     { eyeScaleY: 0.85, browY: 0.01, mouthScaleX: 0.6, mouthScaleY: 0.2, mouthY: -0.01 },
    surprised:  { eyeScaleY: 1.3, browY: 0.08, mouthScaleX: 0.55, mouthScaleY: 0.4, mouthY: 0.02 },
  };

  /* ================================================================
   *  Utility helpers
   * ================================================================ */
  function lerp(a, b, t) { return a + (b - a) * t; }
  function lerpVec3(out, a, b, t) {
    out.x = lerp(a.x, b.x, t);
    out.y = lerp(a.y, b.y, t);
    out.z = lerp(a.z, b.z, t);
    return out;
  }
  function easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  /* ================================================================
   *  CapsuleGeometry Polyfill for Three.js r128
   *  (CapsuleGeometry was added in r139; this backports it)
   * ================================================================ */
  if (typeof THREE.CapsuleGeometry === 'undefined') {
    THREE.CapsuleGeometry = class CapsuleGeometry extends THREE.BufferGeometry {
      constructor(radius = 0.5, length = 1, capSegments = 8, radialSegments = 16) {
        super();
        this.type = 'CapsuleGeometry';
        this.parameters = { radius, length, capSegments, radialSegments };

        // Build a capsule from a LatheGeometry profile curve
        const halfLen = length / 2;
        const path = [];

        // Bottom hemisphere
        for (let i = 0; i <= capSegments; i++) {
          const theta = (Math.PI / 2) * (i / capSegments);
          path.push(new THREE.Vector2(
            Math.cos(theta) * radius,
            Math.sin(theta) * radius - halfLen
          ));
        }

        // Top hemisphere
        for (let i = 0; i <= capSegments; i++) {
          const theta = (Math.PI / 2) + (Math.PI / 2) * (i / capSegments);
          path.push(new THREE.Vector2(
            Math.cos(theta) * radius,
            Math.sin(theta) * radius + halfLen
          ));
        }

        const lathe = new THREE.LatheGeometry(path, radialSegments);
        this.setAttribute('position', lathe.getAttribute('position'));
        this.setAttribute('uv', lathe.getAttribute('uv'));
        this.setAttribute('normal', lathe.getAttribute('normal'));
        this.setIndex(lathe.getIndex());
      }
    };
  }

  /* ================================================================
   *  SignLanguageAvatar Class
   * ================================================================ */
  class SignLanguageAvatar {
    /**
     * @param {HTMLElement} container - DOM element to render the 3D avatar into
     * @param {object} [options]
     * @param {boolean} [options.controls=false] - Enable OrbitControls
     * @param {boolean} [options.shadows=true] - Enable shadows
     * @param {number} [options.width] - Canvas width (default: container width)
     * @param {number} [options.height] - Canvas height (default: container height)
     */
    constructor(container, options = {}) {
      this.container = container;
      this.options = Object.assign({ controls: false, shadows: true }, options);
      this._animating = false;
      this._abortController = null;
      this._disposed = false;
      this._clock = new THREE.Clock();
      this._targetPose = {};
      this._currentPoseState = {};
      this._headMovement = 'none';
      this._headTime = 0;
      this._currentExpression = 'neutral';
      this._targetExpression = EXPRESSIONS.neutral;
      this._currentExprState = { ...EXPRESSIONS.neutral };
      this._blinkTimer = 0;
      this._blinkDuration = 0;
      this._isBlinking = false;

      this._initScene();
      this._buildCharacter();
      this._initLighting();
      this._initGround();
      if (this.options.controls) this._initControls();
      this._initCurrentPose();
      this._startRenderLoop();
      this._handleResize();
    }

    /* ---- Scene Setup ---- */

    _initScene() {
      const w = this.options.width || this.container.clientWidth || 400;
      const h = this.options.height || this.container.clientHeight || 500;

      this.scene = new THREE.Scene();
      this.scene.background = null; // transparent

      this.camera = new THREE.PerspectiveCamera(35, w / h, 0.1, 100);
      this.camera.position.set(0, 1.2, 5);
      this.camera.lookAt(0, 0.8, 0);

      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      this.renderer.setSize(w, h);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      this.renderer.outputEncoding = THREE.sRGBEncoding;
      if (this.options.shadows) {
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      }
      this.renderer.domElement.style.borderRadius = '16px';
      this.container.appendChild(this.renderer.domElement);
    }

    _initLighting() {
      // Soft ambient
      const ambient = new THREE.AmbientLight(0xfff5ee, 0.6);
      this.scene.add(ambient);

      // Main directional (warm key)
      const key = new THREE.DirectionalLight(0xffeedd, 0.8);
      key.position.set(3, 5, 4);
      if (this.options.shadows) {
        key.castShadow = true;
        key.shadow.mapSize.width = 1024;
        key.shadow.mapSize.height = 1024;
        key.shadow.camera.near = 0.5;
        key.shadow.camera.far = 20;
        key.shadow.camera.left = -3;
        key.shadow.camera.right = 3;
        key.shadow.camera.top = 4;
        key.shadow.camera.bottom = -2;
        key.shadow.radius = 4;
      }
      this.scene.add(key);

      // Fill light (cool)
      const fill = new THREE.DirectionalLight(0xddeeff, 0.35);
      fill.position.set(-2, 3, -1);
      this.scene.add(fill);

      // Rim light from behind
      const rim = new THREE.DirectionalLight(0xccbbff, 0.3);
      rim.position.set(0, 2, -4);
      this.scene.add(rim);

      // Hemisphere light for ambient variation
      const hemi = new THREE.HemisphereLight(0xeeeeff, 0x444466, 0.25);
      this.scene.add(hemi);
    }

    _initGround() {
      const groundGeo = new THREE.CircleGeometry(2.5, 64);
      const groundMat = new THREE.MeshStandardMaterial({
        color: COLORS.ground,
        roughness: 0.9,
        metalness: 0,
        transparent: true,
        opacity: 0.6,
      });
      const ground = new THREE.Mesh(groundGeo, groundMat);
      ground.rotation.x = -Math.PI / 2;
      ground.position.y = -0.6;
      ground.receiveShadow = true;
      this.scene.add(ground);

      // Subtle grid ring
      const ringGeo = new THREE.RingGeometry(1.8, 2.5, 64);
      const ringMat = new THREE.MeshBasicMaterial({
        color: COLORS.groundGrid,
        transparent: true,
        opacity: 0.3,
        side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = -0.59;
      this.scene.add(ring);
    }

    _initControls() {
      if (typeof THREE.OrbitControls !== 'undefined') {
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.minDistance = 2.5;
        this.controls.maxDistance = 10;
        this.controls.target.set(0, 0.8, 0);
        this.controls.maxPolarAngle = Math.PI / 1.8;
        this.controls.update();
      }
    }

    /* ---- Character Builder ---- */

    _mat(color, opts = {}) {
      return new THREE.MeshStandardMaterial({
        color,
        roughness: opts.roughness !== undefined ? opts.roughness : 0.65,
        metalness: opts.metalness !== undefined ? opts.metalness : 0.05,
        ...opts,
      });
    }

    _buildCharacter() {
      this.character = new THREE.Group();
      this.character.position.y = -0.1;
      this.scene.add(this.character);

      this._buildBody();
      this._buildHead();
      this._buildArms();
      this._buildLegs();
    }

    _buildBody() {
      // Torso — rounded cylinder
      const torsoGeo = new THREE.CapsuleGeometry(0.38, 0.5, 8, 16);
      const torsoMat = this._mat(COLORS.bodyMain);
      this.torso = new THREE.Mesh(torsoGeo, torsoMat);
      this.torso.position.y = 0.55;
      this.torso.castShadow = true;
      this.character.add(this.torso);

      // Collar / neckline accent
      const collarGeo = new THREE.TorusGeometry(0.32, 0.04, 8, 32);
      const collarMat = this._mat(COLORS.bodyLight);
      const collar = new THREE.Mesh(collarGeo, collarMat);
      collar.position.y = 0.82;
      collar.rotation.x = Math.PI / 2;
      this.character.add(collar);

      // Little badge / button
      const badgeGeo = new THREE.SphereGeometry(0.04, 16, 16);
      const badgeMat = this._mat(0xffd700, { metalness: 0.4, roughness: 0.3 });
      const badge = new THREE.Mesh(badgeGeo, badgeMat);
      badge.position.set(0.15, 0.72, 0.35);
      this.character.add(badge);
    }

    _buildHead() {
      this.headGroup = new THREE.Group();
      this.headGroup.position.y = 1.35;
      this.character.add(this.headGroup);

      // Main head sphere — big and round (chibi!)
      const headGeo = new THREE.SphereGeometry(0.5, 32, 32);
      const headMat = this._mat(COLORS.skin);
      this.headMesh = new THREE.Mesh(headGeo, headMat);
      this.headMesh.castShadow = true;
      this.headGroup.add(this.headMesh);

      // Hair — top
      const hairTopGeo = new THREE.SphereGeometry(0.52, 32, 32, 0, Math.PI * 2, 0, Math.PI * 0.55);
      const hairMat = this._mat(COLORS.hair);
      const hairTop = new THREE.Mesh(hairTopGeo, hairMat);
      hairTop.position.y = 0.02;
      hairTop.castShadow = true;
      this.headGroup.add(hairTop);

      // Hair fringe (bangs)
      const fringeGeo = new THREE.SphereGeometry(0.35, 16, 8, -Math.PI * 0.6, Math.PI * 1.2, 0, Math.PI * 0.35);
      const fringeMat = this._mat(COLORS.hairDark);
      const fringe = new THREE.Mesh(fringeGeo, fringeMat);
      fringe.position.set(0, 0.22, 0.28);
      fringe.rotation.x = -0.3;
      this.headGroup.add(fringe);

      // Side hair tufts
      for (let side of [-1, 1]) {
        const tuftGeo = new THREE.SphereGeometry(0.18, 16, 16);
        const tuft = new THREE.Mesh(tuftGeo, hairMat);
        tuft.position.set(side * 0.42, -0.1, 0.1);
        tuft.scale.set(0.7, 1.2, 0.8);
        this.headGroup.add(tuft);
      }

      this._buildFace();
      this._buildEars();
    }

    _buildFace() {
      // --- Eyes ---
      this.eyes = { left: {}, right: {} };
      const eyeOffsetX = 0.16;
      const eyeY = 0.05;
      const eyeZ = 0.42;

      for (let side of ['left', 'right']) {
        const sx = side === 'left' ? -1 : 1;
        const eyeGroup = new THREE.Group();
        eyeGroup.position.set(sx * eyeOffsetX, eyeY, eyeZ);
        this.headGroup.add(eyeGroup);

        // White of eye
        const whiteGeo = new THREE.SphereGeometry(0.1, 16, 16);
        const whiteMat = this._mat(COLORS.eyeWhite, { roughness: 0.2 });
        const white = new THREE.Mesh(whiteGeo, whiteMat);
        white.scale.set(1, 1, 0.5);
        eyeGroup.add(white);

        // Pupil
        const pupilGeo = new THREE.SphereGeometry(0.065, 16, 16);
        const pupilMat = this._mat(COLORS.eyeBlack, { roughness: 0.1 });
        const pupil = new THREE.Mesh(pupilGeo, pupilMat);
        pupil.position.z = 0.04;
        pupil.scale.set(1, 1, 0.5);
        eyeGroup.add(pupil);

        // Highlight
        const hlGeo = new THREE.SphereGeometry(0.025, 8, 8);
        const hlMat = new THREE.MeshBasicMaterial({ color: COLORS.eyeHighlight });
        const hl = new THREE.Mesh(hlGeo, hlMat);
        hl.position.set(0.025, 0.03, 0.06);
        eyeGroup.add(hl);

        // Eyelid (for blinking/expressions)
        const lidGeo = new THREE.SphereGeometry(0.105, 16, 16, 0, Math.PI * 2, 0, Math.PI * 0.5);
        const lidMat = this._mat(COLORS.skin);
        const lid = new THREE.Mesh(lidGeo, lidMat);
        lid.scale.set(1, 1, 0.5);
        lid.rotation.x = Math.PI; // flipped, hidden by default
        lid.position.y = 0.1;
        lid.position.z = 0.01;
        lid.visible = false;
        eyeGroup.add(lid);

        this.eyes[side] = { group: eyeGroup, white, pupil, highlight: hl, lid };
      }

      // --- Eyebrows ---
      this.brows = { left: null, right: null };
      for (let side of ['left', 'right']) {
        const sx = side === 'left' ? -1 : 1;
        const browGeo = new THREE.BoxGeometry(0.1, 0.02, 0.02);
        const browMat = this._mat(COLORS.hairDark);
        const brow = new THREE.Mesh(browGeo, browMat);
        brow.position.set(sx * eyeOffsetX, eyeY + 0.14, eyeZ + 0.02);
        brow.rotation.z = sx * -0.1;

        // Round the edges
        const edges = new THREE.BoxGeometry(0.1, 0.022, 0.022);
        brow.geometry = edges;

        this.headGroup.add(brow);
        this.brows[side] = brow;
      }

      // --- Nose (tiny bump) ---
      const noseGeo = new THREE.SphereGeometry(0.025, 8, 8);
      const noseMat = this._mat(COLORS.skinLight);
      const nose = new THREE.Mesh(noseGeo, noseMat);
      nose.position.set(0, -0.05, 0.49);
      nose.scale.set(1, 0.7, 0.6);
      this.headGroup.add(nose);

      // --- Mouth ---
      this.mouthGroup = new THREE.Group();
      this.mouthGroup.position.set(0, -0.15, 0.44);
      this.headGroup.add(this.mouthGroup);

      // Mouth shape — ellipsoid
      const mouthGeo = new THREE.SphereGeometry(0.06, 16, 8);
      const mouthMat = this._mat(COLORS.mouthPink, { roughness: 0.3 });
      this.mouthMesh = new THREE.Mesh(mouthGeo, mouthMat);
      this.mouthMesh.scale.set(1, 0.3, 0.4);
      this.mouthGroup.add(this.mouthMesh);

      // --- Blush spots ---
      for (let side of [-1, 1]) {
        const blushGeo = new THREE.CircleGeometry(0.06, 16);
        const blushMat = new THREE.MeshBasicMaterial({
          color: COLORS.blush,
          transparent: true,
          opacity: 0.25,
          side: THREE.DoubleSide,
        });
        const blush = new THREE.Mesh(blushGeo, blushMat);
        blush.position.set(side * 0.28, -0.06, 0.4);
        blush.lookAt(blush.position.x * 2, -0.06, 2);
        this.headGroup.add(blush);
      }
    }

    _buildEars() {
      for (let side of [-1, 1]) {
        const earGeo = new THREE.SphereGeometry(0.08, 16, 16);
        const earMat = this._mat(COLORS.skin);
        const ear = new THREE.Mesh(earGeo, earMat);
        ear.position.set(side * 0.48, 0, 0);
        ear.scale.set(0.4, 0.7, 0.6);
        this.headGroup.add(ear);
      }
    }

    _buildArms() {
      this.arms = { right: {}, left: {} };

      for (let side of ['right', 'left']) {
        const sx = side === 'right' ? 1 : -1;

        // Shoulder pivot
        const shoulderGroup = new THREE.Group();
        shoulderGroup.position.set(sx * 0.42, 0.78, 0);
        this.character.add(shoulderGroup);

        // Upper arm
        const upperGeo = new THREE.CapsuleGeometry(0.08, 0.3, 4, 8);
        const upperMat = this._mat(COLORS.bodyMain);
        const upper = new THREE.Mesh(upperGeo, upperMat);
        upper.castShadow = true;
        shoulderGroup.add(upper);

        // Elbow joint sphere
        const elbowJoint = new THREE.Group();
        shoulderGroup.add(elbowJoint);

        // Forearm
        const foreGeo = new THREE.CapsuleGeometry(0.07, 0.25, 4, 8);
        const foreMat = this._mat(COLORS.skin);
        const fore = new THREE.Mesh(foreGeo, foreMat);
        fore.castShadow = true;
        elbowJoint.add(fore);

        // Hand
        const handGroup = new THREE.Group();
        elbowJoint.add(handGroup);

        // Palm
        const palmGeo = new THREE.SphereGeometry(0.09, 12, 12);
        const palmMat = this._mat(COLORS.skin);
        const palm = new THREE.Mesh(palmGeo, palmMat);
        palm.scale.set(1, 0.8, 0.6);
        palm.castShadow = true;
        handGroup.add(palm);

        // Fingers (5 tiny cylinders)
        const fingers = [];
        for (let f = 0; f < 5; f++) {
          const fingerGeo = new THREE.CapsuleGeometry(0.015, 0.06, 4, 4);
          const fingerMat = this._mat(COLORS.skin);
          const finger = new THREE.Mesh(fingerGeo, fingerMat);
          const angle = (f - 2) * 0.25;
          finger.position.set(Math.sin(angle) * 0.07, 0.08, Math.cos(angle) * 0.02);
          finger.rotation.z = angle * 0.5;
          handGroup.add(finger);
          fingers.push(finger);
        }

        // Thumb
        const thumbGeo = new THREE.CapsuleGeometry(0.018, 0.05, 4, 4);
        const thumbMat = this._mat(COLORS.skin);
        const thumb = new THREE.Mesh(thumbGeo, thumbMat);
        thumb.position.set(sx * 0.06, 0.03, 0.04);
        thumb.rotation.z = sx * 0.6;
        handGroup.add(thumb);

        this.arms[side] = {
          shoulderGroup,
          upper,
          elbowJoint,
          fore,
          handGroup,
          palm,
          fingers,
          thumb,
        };
      }
    }

    _buildLegs() {
      for (let side of [-1, 1]) {
        // Leg
        const legGeo = new THREE.CapsuleGeometry(0.1, 0.3, 4, 8);
        const legMat = this._mat(COLORS.pants);
        const leg = new THREE.Mesh(legGeo, legMat);
        leg.position.set(side * 0.15, -0.1, 0);
        leg.castShadow = true;
        this.character.add(leg);

        // Shoe
        const shoeGeo = new THREE.SphereGeometry(0.12, 12, 8);
        const shoeMat = this._mat(COLORS.shoes);
        const shoe = new THREE.Mesh(shoeGeo, shoeMat);
        shoe.position.set(side * 0.15, -0.45, 0.04);
        shoe.scale.set(0.9, 0.5, 1.1);
        shoe.castShadow = true;
        this.character.add(shoe);
      }
    }

    /* ---- Pose System ---- */

    _initCurrentPose() {
      const rest = ARM_POSES.rest;
      this._currentPoseState = {
        rightElbow: { ...rest.rightElbow },
        rightHand:  { ...rest.rightHand },
        leftElbow:  { ...rest.leftElbow },
        leftHand:   { ...rest.leftHand },
      };
      this._targetPose = { ...this._currentPoseState };
      this._applyPoseToMeshes(this._currentPoseState);
    }

    _getFullPose(poseName) {
      const pose = ARM_POSES[poseName];
      if (!pose) return null;
      const rest = ARM_POSES.rest;
      return {
        rightElbow: pose.rightElbow || rest.rightElbow,
        rightHand:  pose.rightHand || rest.rightHand,
        leftElbow:  pose.leftElbow || rest.leftElbow,
        leftHand:   pose.leftHand || rest.leftHand,
      };
    }

    _setTargetPose(poseName) {
      const full = this._getFullPose(poseName);
      if (!full) return;
      this._targetPose = {
        rightElbow: { ...full.rightElbow },
        rightHand:  { ...full.rightHand },
        leftElbow:  { ...full.leftElbow },
        leftHand:   { ...full.leftHand },
      };
    }

    _lerpPoseState(dt) {
      const speed = 6.0; // lerp speed
      const t = 1 - Math.exp(-speed * dt);

      for (const key of ['rightElbow', 'rightHand', 'leftElbow', 'leftHand']) {
        if (this._targetPose[key] && this._currentPoseState[key]) {
          lerpVec3(this._currentPoseState[key], this._currentPoseState[key], this._targetPose[key], t);
        }
      }
    }

    _applyPoseToMeshes(pose) {
      // Position upper arms and forearms based on elbow/hand targets
      // Right arm
      const rShoulder = this.arms.right.shoulderGroup.position;
      const rElbow = pose.rightElbow;
      const rHand = pose.rightHand;

      // Upper arm points from shoulder toward elbow
      const rUpperDir = new THREE.Vector3(rElbow.x, rElbow.y, rElbow.z);
      const rUpperLen = rUpperDir.length();
      this.arms.right.upper.position.copy(rUpperDir.clone().multiplyScalar(0.5));
      this.arms.right.upper.lookAt(rElbow.x, rElbow.y, rElbow.z);
      this.arms.right.upper.rotateX(Math.PI / 2);

      // Elbow joint at elbow position
      this.arms.right.elbowJoint.position.set(rElbow.x, rElbow.y, rElbow.z);

      // Forearm from elbow toward hand
      const rForeDir = new THREE.Vector3(
        rHand.x - rElbow.x, rHand.y - rElbow.y, rHand.z - rElbow.z
      );
      this.arms.right.fore.position.copy(rForeDir.clone().multiplyScalar(0.5));
      this.arms.right.fore.lookAt(rHand.x - rElbow.x, rHand.y - rElbow.y, rHand.z - rElbow.z);
      this.arms.right.fore.rotateX(Math.PI / 2);

      // Hand at hand position relative to elbow
      this.arms.right.handGroup.position.set(
        rHand.x - rElbow.x, rHand.y - rElbow.y, rHand.z - rElbow.z
      );

      // Left arm (mirror)
      const lElbow = pose.leftElbow;
      const lHand = pose.leftHand;

      this.arms.left.upper.position.set(lElbow.x * 0.5, lElbow.y * 0.5, lElbow.z * 0.5);
      this.arms.left.upper.lookAt(lElbow.x, lElbow.y, lElbow.z);
      this.arms.left.upper.rotateX(Math.PI / 2);

      this.arms.left.elbowJoint.position.set(lElbow.x, lElbow.y, lElbow.z);

      this.arms.left.fore.position.set(
        (lHand.x - lElbow.x) * 0.5, (lHand.y - lElbow.y) * 0.5, (lHand.z - lElbow.z) * 0.5
      );
      this.arms.left.fore.lookAt(lHand.x - lElbow.x, lHand.y - lElbow.y, lHand.z - lElbow.z);
      this.arms.left.fore.rotateX(Math.PI / 2);

      this.arms.left.handGroup.position.set(
        lHand.x - lElbow.x, lHand.y - lElbow.y, lHand.z - lElbow.z
      );
    }

    /* ---- Expression System ---- */

    _lerpExpressionState(dt) {
      const speed = 5.0;
      const t = 1 - Math.exp(-speed * dt);
      const target = this._targetExpression;
      const cur = this._currentExprState;

      cur.eyeScaleY = lerp(cur.eyeScaleY, target.eyeScaleY, t);
      cur.browY = lerp(cur.browY, target.browY, t);
      cur.mouthScaleX = lerp(cur.mouthScaleX, target.mouthScaleX, t);
      cur.mouthScaleY = lerp(cur.mouthScaleY, target.mouthScaleY, t);
      cur.mouthY = lerp(cur.mouthY, target.mouthY, t);
    }

    _applyExpressionToMeshes() {
      const s = this._currentExprState;

      // Blink override
      let eyeY = s.eyeScaleY;
      if (this._isBlinking) {
        eyeY = 0.05;
      }

      // Eyes
      for (const side of ['left', 'right']) {
        this.eyes[side].white.scale.y = eyeY;
        this.eyes[side].pupil.scale.y = eyeY;
        this.eyes[side].highlight.scale.y = Math.max(0.2, eyeY);
      }

      // Brows
      this.brows.left.position.y = 0.05 + 0.14 + s.browY;
      this.brows.right.position.y = 0.05 + 0.14 + s.browY;

      // Mouth
      this.mouthMesh.scale.x = s.mouthScaleX / 0.06;
      this.mouthMesh.scale.y = Math.max(0.1, s.mouthScaleY / 0.06);
      this.mouthGroup.position.y = -0.15 + s.mouthY;
    }

    /* ---- Head Movement ---- */

    _updateHeadMovement(dt) {
      this._headTime += dt;
      const t = this._headTime;

      switch (this._headMovement) {
        case 'nod': {
          const angle = Math.sin(t * 6) * 0.08;
          this.headGroup.rotation.x = angle;
          break;
        }
        case 'shake': {
          const angle = Math.sin(t * 8) * 0.1;
          this.headGroup.rotation.y = angle;
          break;
        }
        case 'tilt': {
          const angle = Math.sin(t * 3) * 0.1;
          this.headGroup.rotation.z = angle;
          break;
        }
        case 'drop': {
          const angle = Math.min(t * 0.5, 0.15);
          this.headGroup.rotation.x = angle;
          break;
        }
        default: {
          // Smoothly return head to neutral
          this.headGroup.rotation.x = lerp(this.headGroup.rotation.x, 0, 3 * dt);
          this.headGroup.rotation.y = lerp(this.headGroup.rotation.y, 0, 3 * dt);
          this.headGroup.rotation.z = lerp(this.headGroup.rotation.z, 0, 3 * dt);
          break;
        }
      }
    }

    /* ---- Idle / Breathing Animation ---- */

    _updateIdle(time) {
      // Gentle breathing bob
      const breathe = Math.sin(time * 1.8) * 0.015;
      this.character.position.y = -0.1 + breathe;

      // Subtle body sway
      const sway = Math.sin(time * 0.7) * 0.008;
      this.character.rotation.z = sway;

      // Eye blink cycle
      this._blinkTimer -= this._clock.getDelta ? 0.016 : 0;
      if (!this._isBlinking && Math.random() < 0.003) {
        this._isBlinking = true;
        this._blinkDuration = 0.12;
      }
      if (this._isBlinking) {
        this._blinkDuration -= 0.016;
        if (this._blinkDuration <= 0) {
          this._isBlinking = false;
        }
      }
    }

    /* ---- Render Loop ---- */

    _startRenderLoop() {
      const animate = () => {
        if (this._disposed) return;
        this._rafId = requestAnimationFrame(animate);

        const dt = Math.min(this._clock.getDelta(), 0.05);
        const time = this._clock.getElapsedTime();

        // Lerp pose
        this._lerpPoseState(dt);
        this._applyPoseToMeshes(this._currentPoseState);

        // Lerp expression
        this._lerpExpressionState(dt);
        this._applyExpressionToMeshes();

        // Head movement
        this._updateHeadMovement(dt);

        // Idle animation
        this._updateIdle(time);

        // Controls
        if (this.controls) this.controls.update();

        this.renderer.render(this.scene, this.camera);
      };

      animate();
    }

    _handleResize() {
      this._resizeObserver = new ResizeObserver(() => {
        if (this._disposed) return;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        if (w === 0 || h === 0) return;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
      });
      this._resizeObserver.observe(this.container);
    }

    /* ================================================================
     *  Public API
     * ================================================================ */

    /**
     * Animate a single sign.
     * @param {string} gloss - The ASL gloss word (e.g. "HELLO")
     * @param {object} [animData] - Optional animation data from the API
     * @returns {Promise<void>}
     */
    async playSign(gloss, animData = {}) {
      const key = gloss.toUpperCase().replace(/\s+/g, '-');
      const signDef = SIGN_POSE_MAP[key] || SIGN_POSE_MAP['HELLO'];
      const expression = animData.expression || signDef.expression || 'neutral';
      const headMove = signDef.headMove || 'none';

      // Set expression
      this.setExpression(expression);

      // Start head movement
      this._headMovement = headMove;
      this._headTime = 0;

      // Animate through poses
      const poses = signDef.poses || ['rest'];
      const poseTime = Math.max(250, (animData.duration_ms || 1000) / poses.length);

      for (const poseName of poses) {
        if (this._abortController && this._abortController.signal.aborted) break;
        this._setTargetPose(poseName);
        await this._sleep(poseTime);
      }

      // Stop head movement
      this._headMovement = 'none';
    }

    /**
     * Animate a sequence of signs.
     * @param {string[]} glossArray - Array of ASL gloss words
     * @param {object[]} [animationsArray] - Optional array of animation data
     * @param {function} [onSign] - Callback(index, gloss) called when each sign starts
     * @returns {Promise<void>}
     */
    async playSequence(glossArray, animationsArray = [], onSign = null) {
      this._abortController = new AbortController();

      for (let i = 0; i < glossArray.length; i++) {
        if (this._abortController.signal.aborted) break;

        const gloss = glossArray[i];
        const anim = animationsArray[i] || {};

        if (onSign) onSign(i, gloss);

        await this.playSign(gloss, anim);

        // Transition pause
        if (i < glossArray.length - 1) {
          await this._sleep(250);
        }
      }

      if (!this._abortController.signal.aborted) {
        this.reset();
      }
    }

    /**
     * Set facial expression.
     * @param {string} emotion - Expression name (e.g. 'smile', 'frown', 'brow_raise')
     */
    setExpression(emotion) {
      const expr = EXPRESSIONS[emotion] || EXPRESSIONS.neutral;
      this._currentExpression = emotion;
      this._targetExpression = { ...expr };
    }

    /**
     * Return avatar to neutral resting pose.
     */
    reset() {
      if (this._abortController) this._abortController.abort();
      this._headMovement = 'none';
      this._headTime = 0;
      this._setTargetPose('rest');
      this.setExpression('neutral');
    }

    /**
     * Switch avatar theme/style. Updates material colors without rebuilding geometry.
     * @param {string} themeName - Key from AVATAR_THEMES
     */
    setTheme(themeName) {
      const theme = AVATAR_THEMES[themeName];
      if (!theme) return;
      COLORS = { ...theme };

      // Color mapping: mesh reference → color key
      const colorMap = [
        [this.torso, 'bodyMain'],
      ];

      // Walk the entire character tree and recolor by matching current material color isn't reliable,
      // so we use a simple approach: recolor all meshes based on naming/structure
      if (this.character) {
        this.character.traverse((obj) => {
          if (obj.isMesh && obj.material && obj.material.color) {
            // Skip badge (gold)
            if (obj.material.color.getHex() === 0xffd700) return;
          }
        });
      }

      // Direct references we know
      if (this.torso) this.torso.material.color.setHex(theme.bodyMain);

      // Head group
      if (this.headGroup) {
        this.headGroup.traverse((obj) => {
          if (!obj.isMesh) return;
          const hex = obj.material.color.getHex();
          // Map old colors to new by role
          if (obj.material._role === 'skin') obj.material.color.setHex(theme.skin);
          else if (obj.material._role === 'hair') obj.material.color.setHex(theme.hair);
          else if (obj.material._role === 'hairDark') obj.material.color.setHex(theme.hairDark);
          else if (obj.material._role === 'eyeBlack') obj.material.color.setHex(theme.eyeBlack);
          else if (obj.material._role === 'mouthPink') obj.material.color.setHex(theme.mouthPink);
          else if (obj.material._role === 'blush') obj.material.color.setHex(theme.blush);
        });
      }

      // Arms (skin colored)
      if (this.rightArmGroup) {
        this.rightArmGroup.traverse((obj) => {
          if (obj.isMesh && obj.material._role === 'skin') obj.material.color.setHex(theme.skin);
          if (obj.isMesh && obj.material._role === 'skinLight') obj.material.color.setHex(theme.skinLight);
        });
      }
      if (this.leftArmGroup) {
        this.leftArmGroup.traverse((obj) => {
          if (obj.isMesh && obj.material._role === 'skin') obj.material.color.setHex(theme.skin);
          if (obj.isMesh && obj.material._role === 'skinLight') obj.material.color.setHex(theme.skinLight);
        });
      }

      // Legs
      if (this.character) {
        this.character.traverse((obj) => {
          if (!obj.isMesh || !obj.material._role) return;
          if (obj.material._role === 'pants') obj.material.color.setHex(theme.pants);
          if (obj.material._role === 'shoes') obj.material.color.setHex(theme.shoes);
          if (obj.material._role === 'bodyLight') obj.material.color.setHex(theme.bodyLight);
        });
      }
    }

    /**
     * Get available themes list.
     * @returns {Object} AVATAR_THEMES
     */
    static getThemes() {
      return AVATAR_THEMES;
    }

    /**
     * Clean up all Three.js resources.
     */
    dispose() {
      this._disposed = true;
      if (this._abortController) this._abortController.abort();
      if (this._rafId) cancelAnimationFrame(this._rafId);
      if (this._resizeObserver) this._resizeObserver.disconnect();
      if (this.controls) this.controls.dispose();

      // Dispose geometries and materials
      this.scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) {
            obj.material.forEach(m => m.dispose());
          } else {
            obj.material.dispose();
          }
        }
      });

      this.renderer.dispose();
      if (this.renderer.domElement.parentNode) {
        this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
      }
    }

    /* ---- Private helpers ---- */

    _sleep(ms) {
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(resolve, ms);
        if (this._abortController) {
          this._abortController.signal.addEventListener('abort', () => {
            clearTimeout(timeout);
            resolve(); // resolve instead of reject to avoid unhandled rejections
          });
        }
      });
    }
  }

  return SignLanguageAvatar;
}));
