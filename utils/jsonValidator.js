// jsonValidator.js — Validates and repairs JSON

export const jsonValidator = {
  tryParse: (str) => {
    try {
      return JSON.parse(str);
    } catch {
      return null;
    }
  }
};