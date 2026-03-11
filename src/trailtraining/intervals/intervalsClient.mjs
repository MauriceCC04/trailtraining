const BASE = "https://intervals.icu/api/v1";

export async function listWellnessBearer({ accessToken, athleteId = "0", oldest, newest }) {
  const url = new URL(`${BASE}/athlete/${athleteId}/wellness`);
  url.searchParams.set("oldest", oldest);
  url.searchParams.set("newest", newest);

  const res = await fetch(url, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
      Accept: "application/json",
    },
  });

  if (!res.ok) throw new Error(`Intervals API ${res.status}: ${await res.text()}`);
  return res.json();
}

export function normalizeSleepAndHr(w) {
  // Common wellness fields include sleepSecs/restingHR/avgSleepingHR.
  return {
    day: w.id,
    sleepSecs: w.sleepSecs ?? null,
    restingHR: w.restingHR ?? null,
    avgSleepingHR: w.avgSleepingHR ?? null,
    raw: w,
  };
}