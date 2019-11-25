These files are a snapshot of active ecosystems on production tutor as of November, 2019,
provided for dev and testing convenience. They should be updated whenever they get too
far out of date.

How to import all of these:
```bash

for m in *.yml ; do echo $m; curl -F file=@$m https://validator.host.name/import; done

```
