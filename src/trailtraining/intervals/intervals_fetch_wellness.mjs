import process from "node:process";

const BASE = "https://intervals.icu/api/v1";
const ATHLETE_ID = process.env.INTERVALS_ATHLETE_ID ?? "0";
const API_KEY = process.env.INTERVALS_API_KEY; // password part
const OLDEST = process.env.OLDEST; // YYYY-MM-DD
const NEWEST = process.env.NEWEST; // YYYY-MM-DD

if (!API_KEY || !OLDEST || !NEWEST) {
  console.error("Missing env. Need INTERVALS_API_KEY, OLDEST, NEWEST (and optionally INTERVALS_ATHLETE_ID).");
  process.exit(1);
}

// Basic auth: username is literally "API_KEY", password is your actual key.
const basic = Buffer.from(`API_KEY:${API_KEY}`).toString("base64");

async function listWellness(oldest, newest) {
  const url = new URL(`${BASE}/athlete/${ATHLETE_ID}/wellness`);
  url.searchParams.set("oldest", oldest);
  url.searchParams.set("newest", newest);

  const res = await fetch(url, {
    headers: {
      Authorization: `Basic ${basic}`,
      Accept: "application/json",
    },
  });

  if (!res.ok) {
    throw new Error(`Intervals API error ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

function pickSleepAndHr(w) {
  // Intervals commonly returns camelCase (sleepSecs/restingHR).
  // Be tolerant: fall back to snake_case variants if you ever see them.
  const sleepSecs = w.sleepSecs ?? w.sleep_secs ?? null;
  const restingHR = w.restingHR ?? w.resting_hr ?? null;
  const avgSleepingHR = w.avgSleepingHR ?? w.avg_sleep_hr ?? null;

  return {
    id: w.id,                 // ISO local date, e.g. "2026-02-27"
    sleepSecs,
    restingHR,
    avgSleepingHR,
  };
}

const wellness = await listWellness(OLDEST, NEWEST);
console.log(wellness.map(pickSleepAndHr));