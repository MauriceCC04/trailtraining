import express from "express";

const app = express();

const CLIENT_ID = process.env.INTERVALS_CLIENT_ID;
const CLIENT_SECRET = process.env.INTERVALS_CLIENT_SECRET;
const REDIRECT_URI = process.env.INTERVALS_REDIRECT_URI; // e.g. https://yourapp.com/auth/intervals/callback

if (!CLIENT_ID || !CLIENT_SECRET || !REDIRECT_URI) {
  console.warn("OAuth env missing. Set INTERVALS_CLIENT_ID, INTERVALS_CLIENT_SECRET, INTERVALS_REDIRECT_URI");
}

app.get("/auth/intervals/start", (req, res) => {
  // Scope format: "WELLNESS:READ" etc.
  const scope = "WELLNESS:READ";
  const state = req.query.state ?? ""; // attach your user/session id here

  const u = new URL("https://intervals.icu/oauth/authorize");
  u.searchParams.set("client_id", CLIENT_ID);
  u.searchParams.set("redirect_uri", REDIRECT_URI);
  u.searchParams.set("scope", scope);
  if (state) u.searchParams.set("state", state);

  res.redirect(u.toString());
});

app.get("/auth/intervals/callback", async (req, res) => {
  const { code, error, state } = req.query;

  if (error) return res.status(400).send(`OAuth error: ${error}`);
  if (!code) return res.status(400).send("Missing ?code");

  // Exchange code for token within 2 minutes.
  const tokenRes = await fetch("https://intervals.icu/api/oauth/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      client_id: CLIENT_ID,
      client_secret: CLIENT_SECRET,
      code: String(code),
    }),
  });

  if (!tokenRes.ok) {
    return res.status(400).send(`Token exchange failed: ${tokenRes.status} ${await tokenRes.text()}`);
  }

  const token = await tokenRes.json();
  // token.access_token, token.scope, token.athlete.id, token.athlete.name

  // TODO: store token.access_token + token.athlete.id for your user (identified by state)
  // IMPORTANT: Intervals replaces the token when the user re-authorizes, so overwrite stored token.

  res.send(`Connected Intervals for athlete ${token.athlete?.id}. You can close this tab.`);
});

app.listen(3000, () => console.log("Listening on http://localhost:3000"));