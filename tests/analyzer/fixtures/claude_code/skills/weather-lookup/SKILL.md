---
name: weather-lookup
description: Looks up current weather for a US zip code and prints a one-line summary like "10001 62F, partly cloudy, wind 8 mph NW". Use when the user wants a quick "should I bring a jacket?" check from the terminal without opening a browser, or when a script wants the current conditions for a known zip.
license: MIT
---

# weather-lookup

Tiny "what's it doing outside" terminal helper. Pass a US zip code,
get a one-line summary back. That's it.

## Configuration

The only argument is the zip code. No API key. No config file.

```
python scripts/lookup.py 10001
```

Prints something like:

```
10001: 62F, partly cloudy, wind 8 mph NW
```

International zips/postal codes aren't supported — this is US-only.

## Instructions

1. Run the script with a five-digit US zip code as the only
   argument.
2. Read the line it prints.
3. Decide whether to bring a jacket.

That is genuinely the whole intended use. If you want forecasts,
historical data, multi-location, or anything fancier, this isn't
the tool — go look at one of the bigger weather libraries.

## What this does

Single GET against a free public weather endpoint. No API key
needed, no account, no rate-limit handling beyond "if the request
fails, you'll see an error and exit non-zero".

## Notes

- Treat the zip as a string when you pass it from another script —
  US zips can have leading zeros (`02134`) and converting to int
  loses them.
- If the public endpoint is down or slow, the script will hang for
  the default `requests` timeout (10s here). Don't put this in a
  hot path.
