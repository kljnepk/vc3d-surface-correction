"""Rebuild the linked R103 surface/CT cross-sections using the original official NRRDs.
Run at the root of vc3d-surface-correction with Python and NumPy. Registered small
coordinate arrays and the two-sided texture selection are supplied with this file.
"""
import argparse
import base64
import hashlib
import io
import json
import zlib
from pathlib import Path

import numpy as np
from r103.formats import image8
from r103.image import png, points
from r103.panel import read_cube
from r103.sampling import sample_ct
from r103.scoring import classify, masks

BUNDLE = 'eJy8u2fPtFiWJfpf8iulxrv+FgTeBx5arRI28N6P+r8PT2ZNZ1Vlzr2l0Wh4XuKgE7D3MdstYr3/8T9+qbJf/v0XCCYx7K8QgsPIXyEMwqi/Lnme/RX65S+/zMOx/PLv/4H8G/QX5N/wv6BPiz4t9rTY0+JPiz8t8bTE05JPSz4t9bTU09JPSz8tDD0Xz8dzBf9cwT9XP0LhH6nwj1j4Ry78Ixj+kQz/iIZ/ZMM/wuEf6fCPePhHPvyjAP7RAP+ogH90ID86kB8dyI8O5EcH8uvAf3QgPzqeD/w///JLOrS/TQom/uHAf50j8m8U+g/Hz7T/cCf662oQf+z+k67/jQDoT85/Vc9vG/CP40R/3ZLfzn+8n3x6/mlSf9s6+J/uw3/d1H9+Gv+Tp//Pev4o+Z/ngP86s39+8l/r+eOs/9iD/+m6/fM6oH9mBD+2E895/NjOfyDUY5YQBFM4CqEIhhDoXx6De3ogHIdohMJQnPrPvzy3/doLkziM0SQGkcTPRv88CP88S1AkjJP/+cjdh3br8r8uZYzgxOOTaEJQCFFQBQURGUwmFELHOELTJEzlEEZSGJwReVHkBAWRaZqiMEbCWZFAJAYjBZGlj+928dL8LhDDnl6EhGKSxPMEywgaQwkSzmMKTyD6+csSOo2pDE4xBCITAs8zCI6hPM7yJE6z5BE4zNW36v96X+ezAvCPZz7/fvxpHsbfFSFkBmVQnFAJidMoROQIgcQ5SdAESSB0CiE5DOVIij9DRQmcwFIcS2Eah0mCQKCMwh9Fv4af/xZIYwRUwESCJWmekD+jwdOMJhEsh9BnSchnXliO0Dn0TAsu0AzHiWcDEiROcAIhCPQRuMbzN1//WvXLGvdp/su/wwTyLFC+lP+t5n/84r1R9ica5vmz1FSCxnlKPvdB1KOcyPAcoUi0oCgYyUksy/CMTp8OEkvTmMQKOkaeMSQQgj3qLIjGfh34s7h0ASfJo+LZsQSBcIIsaPwZGpKjxGMnCUkjJA3TBIwSMBxTJETHaE7SKBmj8I8kGEIfSTCEJXmR0VBKURmV4Uj27CYNYRmGPTZFYY+9PduK4zGUIfizlhSJJwick2hKwc/CYr/812+zzZ+494vLLSzzhl7PIf368XkTi0B9wf8+KOnFfF5eo93WG99SsWFezOuVvLDXb8ebeB9/u5RRP4ZtF8/mVxrDSBUWq1f1sP5xTcueECc9qORDg61zNusXdW7WRtzxiLqPfocyjvddKlhH6nBotlgcz1tx8D7fNisHm3jCrMUf4+VxB+TqFsO0vJvqoD6CxW0gu+7PwGM1aOfvxbhNJJSdhV4AIHuPkNbbJzNdCqScBit/Q8EpcCjhmSPU3/077IjTZDH+08nTpQLCi9Jr2hMqkV8j9dt99hNQ7U9jiMw7FViJMphXDLAgQeuw86F0h1RqIOviowTy4g4Z6jLIm+7M7HMqbo8iLJs5+OjR/Uxy597NzULf0VRTN+SrN7HsRlt6cWHhFzL0AQ5aAHz7Z+qzGNa5pg9yqHXr9aaN5kDFs8D57Ai0TDBuoufR8/yO7WhrAObG2hCW6hTt3wE3MYRhFx8cVAe3B28WJ7Xx0rotOvLirdmkzXf9CHNdZ7C2sDH4XBjEBtOsihcMnsdEflBgUtgEFhphvQHqDYKQccPvCMGThaLpwrkEGsntG0gq58IgoUfTLT156ENteIlGE6qLTcKBZx6UAetogtFf/rmt4807lkmGz9STjvTf/OiCiyGNC6MAkJvxmwXhfmeHLYDfhUVZitYL60GR2AfC4/1+HUTcZbOYZbyQTUQ+wku7OGoRrmde+OTL2sQ5cIKg5n0LwRv48xbr+axr6wbJUXOjzfEAeekUXFXFuhqFib9m5RUZoQIiwPwlSdp53Wtqykg9GgLzSYx3kYfEoiJdlNgb0bl7aentFYv3Dfdy9CpiIrNz+TROmN6U2EbHVZZJDUVLxCJM92iJL5SciCvfue5aC20JHl5Lroq5ExrWx1Bb9OgAwcU22ONLd5NNwUdPcyGf88C/Hafrh6pKZ4OlyMIWdM66SRHINkwRz66Vbp0NM5LWtat3DLurVHh3fLPrkTMVr/oEYw0yGtrdCoQvPfTtUhSZi8LahDWic0Pm1ANMRBwB7W8tnXm6lGNh653Kg+JyN7jvfhOZlYGiBm7vq0XLxMnhXsRNKXHbvjAjuY73Zu+3op6d6fIJD+5xgNCh8QsHWnxBlBwdBMyJerjpqrCaJ24up458LRqw6v1SqktgV/vCm0XIklakfEEQhbPDcLEOJ1NhFyP+tGOdvytuFeLTdqSzkBj9Mg8DdwVZWMfga539IiiVrWXZ2wFE6nsQhsOGNjySjCMbjhLDBFcv18YUQ7iXkppl0zJ8AXHphZhdWIKUsI2zlxjjla14h7U3I97SPanDCKeaOyNZ1aYoXNSOsDrmbMcUqFUtQbywyUJCGNVwM6kSXoTubIJ+8V9+2FspFVXGLmFdahFAl/h75OLja7Acpa1OlEzuniJCGMt2y9kCbS0RiFC9gIyEiTbiGzs5ZxDkHladtnky1KmvtoISWso/chCnfh8Dgoh6smYqS8u2UZJOr0FOP6Jr8KoGLIW+eoqKrXjdRZ/2Am/V8r5Ccwp7xxUMcw9zEH+RXlpVJretieSJ5d0ljmupsTT6oDw6o3HUonDNhpmWfej4Xn4xwJkOfYRvbN1iiJZQYNtppTUdYSTouWWD1pdHFmdNSgM/udOQlbWBAx/eknPlFspJXeVDkK1rVVuUlUqXm7n5wunoLdGWkThtfcXH5SIMXOKonIiVgk0+SHdYMh75GLPTxAtB15vSfVtXVp1MPaK4wuOwG3B8BdP15t1IDdiG5SqNV1xAkkttwkdHESXmtnPbZkXaKKE9BgQd/ZYofA6aBYQ1qWeLGE6TZe2DM/fWM4sep0812mDS5LJKCLwEe6xYD15CnibNVRc3vSScZTXlT7C8pbqp1TrpYWr7GIPnECWDpQDIe+/LX3bzcCR2f7kpoN3cyTZXQoYuZEJZZYv0nJ+bcrGCQwReJAhmavigKFpRXDm7/eHuU+zH/hO+kWMLzXXkI5fKOFJSjAJfZUEDGrWFpojtLSq6ozB463CqajFqLQv83LF3n94iOCL1r8NGeFMGJs7RXu/FSkKPq0D9SpLsy81YD69eoA0j5bll14wmY56lxHkk2Q47H9qSSNpzdY7MoNdjI4/HLm0eZIou/y75oLor+on5Altuu27lsim8K8mdhZlwpRu2Zcg1Vlr4zkObmq2D9qp7uOh8859iBaSbotVN4TbTXNnjphT1PKrvBgYM6mRleNOcBIV4IYGqT3UviuWuavK8El6jOpXgIQeorkHfaxcMIaZCXXd5u7Zwjb0X9CTncKlFfXVP4N4uZ4vleCJmVqEBKrrJ31KviuLm7mqDCh6QfIJib69J3iTPrSOrZ8hV3K1J+NA49KkBne2xerY3kt7S+x+H0x3F7kknvVOZN1fVDkA1bNiGOJ4I+3bDBPb2scocWT3egdifB3RReVy7o7o6m4TKuOh7y8wDNvY+9mBBPVSAkjDMN0cl38OG0erIvGSuT6tFjc1lKBNQcOS7o2ZsSC0OYJ6o3VGRbnO2FxhryUIDhnhg+3GTpm+W4q0mRwa+nY8Nt4koE6kNKk9YYql92Jf6SZ2ceRoHbufb5b+SvMl6AHWbqV/17+VJeAa4jYWSgVbjAs7vph1MqYaKUCXXtZn2KWTJrU2g3IpjDQn5HpQEcmGyX9kAe8Osoa9WW0frj86Gg+HyKfGngsoyjoKoF5lTUxSFN3/DdEnPRZswwBZ6Rnutoor7HyQ+erV99VerGyPcOJJgdeKKX8lhSYXV7hZ6gX0/PAlmsz9ojLbmW6/6oPcdRQvv04EnUnxcyik/J4ArLP09VIT8boy+pFciIXMFnz50Z8nRRkC2c4sXiNskF3nja3v2riVa7glaZcyWt7Ct8ZWL6nCVsb2KPWgg2YIAOH1K1ciZEqU7qMU+GidnQbiuGj4gPwtAfiHy1iJewTZrPc0F/zoLz5JLuH6dF0cwGy7Vr8dFE0tEKQ5rcRyIBXENotFGm5kJ5Dgkn7gwzZ01qbirBtUCrq1z+UPMHrreDRvuUdMdHk+R4UQfKTjOO815idt11EsM3D4PvD2dZ9/SG57qEn5JJGHls+FEtgUhr++GI3rogXqn7y3SbvMWe5Ilo9CEYB0xnRV6PACsl2oan0YJMrglSx6/TG47m3vu9O/4RjzdwA4uSD67KKQ0+kCNBrt1mN9Xch8b9InAG9yAyNxdrZMFJctbUoK6XybarO3d5DHqfEU7uKjbnc/ASWHnDB4jqjQOMGm4rgcPKcvHf8AbmBvUWHx9hRH32nMPoA25JQvf3VIZTx6zHll6Ta+CRISMNpdNWKflSUSpg9th4a/+/imjWSWUuNbXz5FLJj/NU+WtyBJhvBAu6VuAeUtdVpKhtr4KCjqv7sK/bGqVGgqBVfnope1YruSBqnXtNNDi7EI3btce0nkgrMm2OMY1DiQ+DOCVXqOkA3uoU4U2TpxC3k6QPtn/VGl+NvHKVNsYizRVb9Udnnl4e8BHdqslDMyf8m5wF1e1xVhhh9FhIzDAzs9R+InTqX3QLmxt5AT05IXpykEmhsdkNai09f5Jy+gS8hVPIjt6KhwZTG+o2Cv7VLVdwZgdDt6tS5ibUJvgmg+4njFv/EjnuwXWspUNJlwIQwH9JU8Ug0Zz1awEsvn0SVh36XyqhGcb3XHOCOi/+iGZrAJPxpShPxuvFhf37ruQq/l03nVHfs/vNeITfhLwdnODkuEKGPtcOjPsBRxcsAU0S71buw+pkRUOKiaihWwM0tSgzY5Q5rvmq64q8hmkEtadcwH6ZDwJ36uUEInYBM88itlMj03LdLrh1MP80OXXkmi8/UYu0Jd3gSl35l45QRPae0VAzgHNuBceDHHohYMst4juUpGW+f7ZL+bUslW1eHf5pmBIeOYw+v0U1yFGxtdOBXK143jBVgvlAmBLUZxvDEv+JJ5MxZVB8farqRtyxFtwrJrFmd9HMcTBXeDIXhFe/KAKqm21d71uiFITV7q9bjqpfMFi0WJ7hTbVpIIi9wEpiucLdgHnBdqD7p3YEcuChb1dxXwQ7UcoMa2E2URz1px7vwnDK1TWtgwTAjhqbLaZiE9+he/0HgZUhisMM90ZToUO7n2sy++1R5cPfFDmIso7pv4R77P0LUn7P+N9t+YO7fozvM9af4f3g/zB+3Ty4H33NKutP7ku2hL8FZwqs2I7ZAO7b3STh9kZX70vSUrL9gGeu3qzy16/ME+OcPYzTErZQZDdyoquT5UbK9W3U75Y+RQrkahiL9dFVbCbQRPtEq+bU5CC6/2e9cemI1zOA5BvvhRAbHaVOjzGfP1ZXyzM6YOn6MDPXcnRDSiES67DTnutIlQIzCsAdQfbNf7wVC3yukOpv0vQuw4GoBvu7DpMP6PMnJwEZ7hATxm3dv2bnaXnvHoav4pWogXYH9EUrBFP6o0T+uJ5/hmKiHiGiJEtZqMygflXtMNu2EvE5r6+E+xMGd+hNcV2MIQFGfkkFw4yywkepo4yw/xKGMLhcvcB1Kk3xtOWMCqpOrId0lemPXa2FFWXOZ56O+4lN/XwifsPuTTLWy2AKl2nIumY7Zs6I9rLaMPbYxpod3PTq2ZpNDei2zAAZe3hZ94TfYco6zMHbSO8dl6dmi5oOp8cGZjduDfRo/9GVtdnJqHd0QYLXYqW2fcEja2azX52zpTDZlFZvBFF4MTgtQmC5Q6PT1V9+UoZ+rr3TJx4os0TSGMejt1drCidshyoI/lEtiPF6rK7x7NTmp002dlr+xSJc+PuYkAtnNgomTs7bPA0yiVaTi/h4l0PUmpflgC1/DeURpImiW9Y8NQ9kON60ExkAW8jN+U4CS+p4jLUQtTxfa4Pylw4lg0qGb3xylVBHXU1gnydJmUKYOzG6rdGadV1REKqIi61u3RB0n0KS2ctHnh8J0aMpn6CjG/7HNhLuZdcx/mwg4oJaed874z6VqX+kwEaHuKxlI8kh0juFaypxrzWYoYVzFXU9qmMGnPfU2vJPqzVZHbgoItn4B/ND8QrcOhI+/IlIY041XU+n3xAjg47awxYBV4ijgO2NkxZNzJRNx3RJUGKtOmcHuRTIHtqJXB9BMbxrB0XpZNMgexx/UmIzcfM6CkQQyRgZs9mUFmXrnNBcqjagtvFv6c2lJt7fYjO08Pphbe3vQqgAo2C7lsJBzrhTESfZesMVgb0gIXSz3RflUe0co8tb4TvPOzqvpAgfxsQeBIAUg3rK4dGnSlXT8eDAcClw7nfd7vkykBqth8YWMVuTi2fWi14n04bbDqNOnlK7QFwmuDBEIVCGaPXD4G4oN38wlFuwttLRpJVz9W8Xfr8cJSp+HrkG3FTtxiJ7UzoDFQS4+xXE9asZwBqkyzFnFgejUJUj7Xcp3R1zRMJ716FKExseU1mKHEm8Ina7pMBRsMcvn18q6n7QKXcQdv+M0GV3ZX4iO02klEF3B5LGsXCqUcIghx9aKrhHkNnN624GFjuzwubMbCUK+BgW/PHhAEj5nusii6ywOcoLjrMP7BQlpA2mC9xFVxH9xwUr58KPztrDYKorlWZeuc36wvemDX7R6LmS0tp2fxCs1BbN9hBTENo+34V7EtbM6HesNg0blkS/KkvZkHJzqzH+uHuOCNi9ohqoVbpKQjyygVzx1o+QmZQ1mSaAq9JuyptvyjtxGWiVZ3Z8KzQ5XX6LPIXRj4+UByegpv6djo+m0nX5iAOdMki80EtTeXq5n6ZBrzWnlgLkSrM2VHRDPRteqqpPgxogCPVfCq8bxL3KpvGrPVzvny1Tm3YUZ86sYwWVg5CTS8ESg+fet/3gb4Tv3XPKFujRtobqpFZk2qVO4v5EpwSjiwJgTj86rC+nfRW30r+DdzajcV0syAlzYSKH3XTuPQLHmiJMXcBtyLeFT0bfFXo139n4Nne3w7vq1qP1n2jei1AxO+3f+/9Z44vb8r3tOw66Qsnsi8k9HZ3ThDfaxp47J32kG42Tta14zwGUcA2GZ8QouN+lpgWZRX0GHGOU6FXwgcfPwmt9MNuNZxPSIqcf+iaKNR6+IzrZozs6wOcTdI2oUaCTceqKMSPADXR1QrFb2aeZdQOZs7lt+ACTqqk2Szq+Ld90YLno68i7jyvsLH66LQolNNIjo9i7rK5+QZo217+jZuhqTNcKixb6CsgKS4aIdnJqd6EDU/WjD6K/YLFyEpgDGFtgGuhBrCFTI5Sy2wNe4QvC1ekwNpScSlhoQ8A9mk+FlnVBbB07jQjjjOf91G2AmzcNDk7U04Y2hGxbok1EoWmnwVfWE7exE9Ya6g1wsGeihaMAmBTjw472ONczDJNi8G00GNAbGaJP+NOXO4+bo4GHTSvq8geOhv4G0LXG5nvKhu++OoNQyZYN8iMXI+CSfVASgcjz6gHadatdcwiCwX+pPCmEqBOocOEUhmSGhKB4LlG3BDJXnANZMn7JaBSiCZZKMiR99H72pZUxOdbdHLZZmm/1LZMA3HM4ZPczEuB/KuE+IgDbEAk4srLc955UzN1BW/s5t4F1d37zUnO0K2M9131OVwqw8D18K2MSShY0xCtOfOpHVKHdTeMQ2o9WbF3SwHFbXnjE2nL/FMV227cW0v60BsBM15396Vi2HEN2VF/9OCIuVo66hHwgEecevbWi0MurtmF9D+kEybJgFZgTc2jlioGkHEmyU4fhmNmLLyT6M2/b8pd8kpXODyoTP3Yw3JOZv6ptCINUbTKlMnJN2EOZkaRC/Ba/sxBp9cP6qmWopbrCbuWgp856nvxdQbOyJtP286pKTuLeVPUyNpDui0bfPPUtcCAsaQi/KIeU5ZTxyayUDpTCtc9JawQtbl40kHGajbuGf1tdB6XnplF2Ojhyr0qYYgSW8oT5Ib0gb+qplyqb3gCSFAO5rRPu0zQzThAGiMZtlGwx2DDEaykIS2+33zxgFPnb/U1tiehgUgqx7leTznehClQc32IEGZ9l2P7Kcl3YEmqcZrLKkfHJ0brRjt10KzwstGrPQSeykT7PKGgQe59ny9Gz2VEykTLE10oHF/F/vYd3ozWjXHbyLMSkWhHr7jxvMrbTX+y7yXSsaf1eS1umvBGni0sKrOVy1YjBXCznMKslUB8R8YafS6lDRjwQ9GDi7QZE835Y+Ln0ZfvniJblDCk+6p7E6ePtsmUyEBX/FZXQj2oYpvvq+0B5IBjFG691fYfyCOa7narzgdtb7NSw57geYF9QsNZ9QcmQ2xjQSkOlamiRZqOs9iE4zon0HprZpyOZ591zQzCkR5MF0lbsvEQcuXAQgC2BPrVSVdVasKUHOzBRDeJrJpnUQXMGUwUVd0Uyl/yAzK5rYW6J5J56Ya0pUur8jtWKOep4jNS05N3M4/y7jjhtgv6F6B0ZVocmtiWCq6NtRrQ7CBbwrUieAynFK6+AKIkHwqSPA6Dp6GUG66bJlt58hOguEzS4e8BYLhKu60YpnwteDHf7W6psNa9WO4bvgTt85wB2sM+FZomHo9rPenZVxJrBbhLGEjq/gI0bIr7jVIl22QEGiaMT+VtV/2E9nsrJUtAnBgaowKMjK8nFmnYFk3J7gcAH3BCyyR1IqZHlQdY0xKehj63B1Z6NvLOIzqygsJm6UK5AmM0xcN17+CUZ1f4hQNTkzEV2stUCPWpAJRoUSXLjLO54hGfubL8gc8RgTISvb6wzxGtQBib/chKFZuVUhrueDsPBUBG65HDPUNMi9LdEELlkyac6zdUZTLKSUM4mSLxRfL1Uek18yyAX6IJsvdw2TYMNOghbtJWm5GhHvUtja5p8WR4PYidOOJppL6uTrunv9yqIq2hFwSa1tZDXIo0YEKEyQU7cYarG7zheiZRDunpTycrW1HTHL0fKutd0Eq+UIAQlrwDPo2cv4giGeY2aFP2pR1oykpl7cyGh5GufjPlFmm0JORAx+x3JgfDprPCtkqc/KKcSGWzm+JDNh8Wz6N50CF3/1DJl2VOxLnp6OKK0Laun+yJCkXdY2GxdvIWNpoqyp+CpQV1EmRTxYwjy25+suHru6QdWtryYvkIEH1B15B53oFmRet5MLxNqvHhrxfnrIN8cGTjGzx4s1awYd9I1N45AAoumvhH+v7M9nfGxuamri5wFwccDMoBVBbVgNmB+fLNQV/UrILUA+47ktybdgFcu4m8I9pOeqKP7KkQtcNXljpavhyuG29kjZ2127v3AgPBFrn+RQ0NQxeiv7bwp0F69HlMJMojIXwDPJiPfXBv9aOtv/6S/+sH99pmZpfUQX29mIX9/q3/747vi/u1/Z0T0P6pHDC93irN/X/IEaGfdwm/v2v4keMKv30rPTcwy88VeKa/9vxQ0PI+y+e/lvFS/h1LI6dwisjQDKVjEikKmoLIAsEhhEChIoFjBKLjHKbxFMPoFCbglMTjOM5ThEYplIiL9HduBYETKUZDCYXnv5IlaBpGYzxO0Uce/XShMQml2FP8I0ScY1RMoRiOkAVB0giCkgjxy3/911/+RqSDIZj+XxQ6CEUo6DciHfw7ke7/FcvtL8iPDuRHB/KjA/nRgfzoQH50ID86kB8dyI8O5EcH8qMD+dGB/uhAf3SgPzrQHx3oo+N36txvfD/oTyle+J/Syv7Y80fi1x/JYf+3evA/1fev9OB/d/7x2z/O6s/Ja3+kof2xB/+7848Etf+znn+N1vbnrMU/3In9agH/iwCH/xjhD90Ngh9ng2H6L9DP9/9B/hj2048iCAaR+KP0p/9PGG4xTSY5jhU09PjZU9VBOU1CGEEnNBonWUHCeRI/jphneUqlP+ywIsFxEkUSEsfQp5D/A8MtiwsST7IMTTCyKBAspqAERvGYfm6HHneOIejRR1BwQWHxD1cMItMsRXHy8W3oJxb8M8ON+AsMIz/nHzhu1CMkJuKUKgq8yLHiiQlxTCRZAuVxjOREhhcpnFFZQSAZAWWP1rzIchIu4Ayl8AL7A8ctSZ7byBh65OZJCv0Q+gqIeFYQIxAUh2gS/WHS5RCKJuijM0dS8pklhGQk9UPZK/6U44Zg/zuO2w+LjMDQrKBhLE6INIOyNKepJ0TCMZTRGJ7nBEHDRI5DMQaTT5xNKZpOkjSLn3Hl6e8cNzSJkxSPUYxMimeS6bNNP6w/Ki2QZzezlKZRmKSeJXgs5dk4+hktRlIJTCZEjOJw8nscplMyReMCe5aIfuJ5jmRJlmIZAqdwTj8zxookwWOMQqgsx/IsRQr0Ccf4DwWwIJ4N/P/huDHy/Ea7f53jxvj/8M4bLTX4Qc+UW888yu984LpTPH0/jvc2TOea193s4zGZkqHR3V5zWAW9Rfiptc5Oa4SBgGhOEdB43GYDsl8tWbdrrFhMLJRP1SLZmiW+/SIqcAAsTI2ibmT3KbyjozWKcxQNgjo4oaNzotuVoW8eHQ+mjpmwKz+U/j4p37hflFQdlDf6jRNNe69kil8699CjLviy9FqoA1Um9roUihzyBGv46NWnUkkGsqIvShYkEgNnPAi5flEuLggVQIVqzTWBUiTC/joyW+L1nubXCZ+CQA7aspKK7viAP7/LRXmCdBaYU9QLoHuic1Eh/tZjC1QkJxZywkEqjlkvXLac4F0LY9ghJ1hJa103JLpOsgHv6NiIN0cSOsrIwRrsEsz299p4uB1d0Ti0K4x3o+BEzhw1EDhvceZfaGhCtLQZMvGVXsVs9HOY0u6e0xnOEzoBdJ9g5NrU7cfGw05o2eyXbG4gCvOxAnCzk0jeCp0aoRDO7reGtg60srY8jlpNLVU14GjBbUdkbr7VAIyTO0Oa0pxC5T5OB1FlXaKXMuPlBrpkfpFk6EMCJGFl2bQlXk9EYS7dWHzK0eygq6qoFG7KT/kdSslR+cNT8FnnbpredeZpSbUbSEo5Znj9isf1QfGTDAlcjiQGYtD9Ezq/HcgT1feiA6TzekfNz1ndZNMQdhav8Iz7zgAgkdYgdVNX24QQpEYN19fpLQlU1yjHmSBubi0KsxgS1VWUwg4th8iC8j7A5d533/fNalZorhatpJY0YHxsX5YNXNZbGO+m6i7GRtedVeIJXbIKq1807VcZUJwrhdLmUTuUmJ9An+55aIWAHaAJyzjF61QxVyfCWzxzHBF7zFtrGNa6y7BkUapZtP2wpqv2K7rI8P4e7pST+xf/2smWnyMRPyB3t9IrHXoicKZoY1fnM18EXJRxFL+NNdwNwA+7OyR31Vp9zb8EJkO/NwJxFTnKExfKIP+KRILCCSZgyH15R+aiag3CVc11Z1GsyNfQFjCf4no4QGNstj1f/kBROXVrZS1O7JMrjxUS/mPg6gpuK79ThXpjqMf5PEwJwft27pJcOKdkThlosibCJgsfMAh6HrXFbL0kNOOiBbbeF8+uiZQM0AkpPHONPpddBCSgTWlDKgR8T3Qfoxd4uJQZZnCmkzEczCF9d+8NhjxijVbt8aRm/gL2Iq2abS99S4yr/KoTDTibLjxTfuwU6oxtoCtb9SxbqW4D221ZplsopiF5pf/ml6B61tW6dfFxApnm6Svbirbf8HgXfcehr8VnfYSkjgGtT2tIJYH7dB9kN/zWinPNwRIQQZo2DbcP2oTNsOIsMLYJUvRcoerAOqrfsqLVllkkXxG/PsLNWMVYttmTFgyHvq6Opxqgz3UMBIZYrfodeAshbekIepBUa989MZQOSHwjplSXzvqhE97Np4trNZzAaYg8Opwk+b6xlvnASeeHN8QD5fCOP86DFJTssIrMleGnmIfQKeJBtE6MEAvCa04nzSWrHrmK2OImc16OdsxMdtgHqDyaPUD47hMLdn2pbHBMCOCvfZCPzWAQKgfjjF5djZCJA98eGq48LsWxZnzfQWlee4or4KFt3skuuR3oN4hQMIZ5BWnDVS07Qm9E15b17D6FaFCKFIIoHSzwzVf7AoFPfcO588vFTFvuXAEXlIHPgZt5WQ+4l33f+fktisUtG/DETianlS1Bzo164gL4JUgniD5A5mkNKAxkFQBao33dzhV70T+HhEr7gg/S6NgDYW1GrUIFP/hePrXkYWDkTRShrS9r2fqmPQ5c2WP0zXVkSR45RYqEF2gONzrQUbrN069OpqaFDD5ZSh6AFbHvdEpjU5Db4cIiYh6PWN+l2JGzplGsPiCrOz/QzTvhKh5j0b17Ip72Chv2eTKdpvfyjx/kr8bQcqeq6ZnJ87BXHZx4B1lIGYsCW9iIbGKvcAJqZD3/bagwkSJhSU15rWWQPV6a1wmwEjUFUoGOQXYFh/tfTL4JXI929+Bdk8sIWD8LSqk2fUhsfIbuO7VB4XbBGmUpP+yLosrlWo2ptOmaiAS4CGtWUw7lnUO/OU5KK5QKOSzG+4Foma+1FEq5TijWTDmvju3xV++gx5Xhsge+yfSKzRq7+BM0T1TcUoBa2TK+gegYwClg6z0PerAoCTGz1HGYoqFMad6IOvSTy+/MInALgJa9Y0UVPlAknULfjox3lmt6JEYWlJ+U/K2Z9eeHKxmok9rwbOSuA37Zx8Kk0OmkB5KwLRvoN7NK3kuhY25P42I4MyY81lNxEzO3mdCS5rBhll5GKudEpq9BNBW2LKpVrp13RhpayaxQLvksUnYY1Nw6RYwU0HjzO0XMgUJXl74LUknoxKH6ZIQryvsEIEZekPvUtHffj7rbsTeg0NUFvlCf7xQgGWlWW3WR6g/STvCrIlH3xkOoO7+rucsvzXLmqjQ3E161zmrQpGr1x7qagbRk8/IofpYwH8dgJPYrbqUvGzA0VaT7N5bwBlqj9ZDu8GsbCKOuP+jiL9Phf9dVItCFQz3IBu7BuvqxHY822wTnS7ncU8JnvZMlH3C4ksbBeVSuRZcLwd6uOPq0ulpDGkq14aaD4ep8DcWM7QgP7uzcD+bVbHzNUDu8qODWfDd1+lywK0/toHv8MUeoJnMEqXfMd0ZzBe8/cpyTivxWOgWumcu9RN8iSa1rEtijwgtwtY+5wBBHC13Dl+75uFf1LH5xUlQQOi5BMIB3dDlvBZ84RAzuKQHDbyX5Wwo37T1Jxnu/Djd5W+0Xy6dYak9wFHRW/mjkKO2jCYPM17lU2mJ//qdABKcfZb3H5LN7VRoGmJ0Vq/YugMsA+6kLgcDBra9Kvm5Pzt5EwtncClzLLmDVpYv2ypoayD8pMjtLaP8ErHyQasWjQ6a82amMsrRz4/Wy6TqXsCb74DImsWoSTb20jftrnk+bbxD/UpGs3GRJ8fZZTBw6uqYVhH0FK5dkHLTRMrCgs19qcls3RSebWw+mTqpHeuVCH/sQqkDBmoLKKtPK6ZUjwLqVH12svfaYqyGrCzULj/c3586EMBafvKEiHqOwPNl393uAbSbi7lHgaGsJhZcLIUl4fYPQvYkgtSZRHCaMyYJTQRpL1tdY9764ECC8wLaqBjLF1gs1VEUnIkY17yp21kTQiAyYEOaqhwiUeP40JWWimy1LrrNP4bTXR8Vdfl7hK0H8VomoCn7IdkoTwfdnh7bxsrcwqjKyknuYN1IhOflBNd5t6jek5R4IPzlJN8qh4xNn1y6zFJD4tX5pH6pw98KpaoGO4n6AixXAkZ92s/1NnQeAXlUM2M5FJ0pO1tZlyPTudnlPzrLnYCOQCuxQx9la2lfqMsZAmG3x6jxagtSwS5rgwbrXHG1cvay+Ak5r5Ls9RXwjn1z5SmsLaxfSfNjsseumBVPMxGoUUgVY4qxqCuDz1Jlx8FO+Jz9jHXsJO1NIJRIVh64utENVYqUlSUczFrLy4MJjxGx6J8M+onlK7KGnSlyWCPKYyk8KtW/1cV7zuoZji3LHzI245nrdPoDI8KPIanANGSS006qVUs1LaWCW+yLLz6/cT0aKyzKbusCtT4KoJ6gk2qZ6x9wmD8iCZVjDP5WlmFtPTjTB7JQGX6A79faojU8YQMSJoqSzg67mJTsW19Hv7A2RG1yHaQdFGpYE7j4pTZHUUPlNg3Sjr3swRFO3+DZQL5yLkZmIj5kHVtsyPFo3cF++SXheEDw5oJYTBweejbySQijWJrl4ywBNXXZm0qwlwLwEG9Kz96cykpb3RfK0CwzDXwh6/iqyzFTk4iB6dx4M7SNjBMLYJDoL1zHv4noz4rvi19f+o1fFp++qxhILtIA8AgQxJC7E9vqFdkZlzslHflvsHVUXUR3J5H8DCXoTDSTfXPHr+2SNUz+/vd19jvKl7TjD1L9D7eP71POMrTmvW+3wI/9wL+5v74d/PbSf98W/HX2W4Q9s6oGFOd+5GsbY8njjaDJJ4D2ZDZdK6/Q4YpJjiH5nTf8AJNkATaNtGqh6Te+nsIw/34xp2Hg8hdZt2JHhFLtxHsTrzgL03XUI3GcaTFf4qThpfCa3B0BQFH0/G3w5WgYWGSX2zoJWXdWIMzxkuyeduaKKoQG+tNCgzVcoWlVZ8drOtlA4eBMKGABxm6xFrGEpMhdgBm80Z8+GRpa1/o6FGHPUzpKBCALbXgctL6t628V89GThlY90GDTZF6myxJhaTxww2QNOOwreNNJkFeEa7H1MU7SW0MoiV+9FBHwa3pGbpweE12Hb0KY2lzUYzvZ70cI0kZi3TNHuUzbd4xLbQZ084nvTqNsFCZ5LcJCdI5PdzCynasgZ9QmhIo1bzQ2nBGkmu2Xit0onJRN7T9DqtIyvd9HVAvx8x3pkSqRAfgEGUQ8oA5uCiyIxwcJpMjqAE8oTNGBGRuytOEAU0WuPu3POsA9kXeyiRirUwyLo9dbStBZNfM/tM9Lo8MKlQJt8iY0uyC9kUqEF8oORZnYyCuno9NTFS0B/PlKBV61ERd555qaV4pUarUZ8fD8PcualNe25ALAwUmfu7HoyX+apzYBYkuPgYa9JnfuOU9RPFdQ/tTdr9THp0y9Ki8Q2zt9H9H6S0GmaCTW7wSrfl59UmbBFpePFtPrAMV6dECHzwtMB6Ou7LlDYQ0aXVw5oyqeO2y4vYkGSx/tnPrY3UHliIc8CoX1K079Gehv7Z08QPg780RPSMGcTs1RCoE3SlQ5vBb7Xqxg6FygIrq1R1K07bwKM9ZOjBwZ4MV83vWbmzCVQOxQo6Hft37MTT5MFJU9KMV2y/fSU40EVQ8AP6L7plcN31yXXvF3ATvvGb9NRr8G5sqmSMd8IZtviyN1xZJbu5jGvqGxKUM1pKBgNXZP3Oxk5BRPHdOgYpFbXNfdB9ZvWPObcbaUpbuE9RKs9Ke8vUqOaZFwCoODwFHw+ZG160APfymWcNOTVfCKCXo37CaaCQKLQM3Mb5jKY2Qs2KghqWqShUK69rjoHQ9tnCOIQDq2jewLFqaQ7KjzsVl/2FD3k9lhr8aAP6j/p8/O9IGCvef0ool5tldOA9yZWLPbALq5q7TxBMOiOdWAeEU9RVfMUl/DWJ0Xh1UgzeWdCzHCuuxpxPjgCyULGpV6PIVnUGH0laPMP5KpnZcZI0a33E25TQqbfg45o04Wo2WzqqtLcofJg0y4mnuqrN74YcLmKEwMs+j1VVCDlQc6ic5doBRZSfykaAtgJuoA86uaktX0KrFgK0EoN1PkzltiwLIERj4JF2bYxOwnrFOPOfhe5D+juEAwlODYqeJ01cAH+JXQhCxsTRl9IZO6hUEZD56ikMbL4+/SIbuADce/rCXGws4emdUHKyxLM7KmWDZ4j8CmXpdDetY5KaN1uJ5CH7H6c4SXo6PIJ8TeN+b3YkEwcWkxhhqk7EImRv/uQYBz2uxXC93va+kmueDpXiP2eYJ/iwAbNge/Qn6zpy+KNBoTMIAptffcCcHvWD4SdKkOcVMNxs8nNyzLKRM3OH2t0s3qQ37cxYZ55++c3iXb6ZpXQZdEPZ88VWoGQrltqIhJkfcFqZl7pDCRwh8ILaYPIJ6O4J+olNb1fkahsSv9tEyabYfVD1EpQAY2RPyGmdpw3nAQmDmveO1aPKCTXFOihHlj17QW1WfrtJN0TUVfK9g6+CQT6ZPvIHOkytrMCGFHqTfJ3Rcl4aks/faPe3oQjEta8vR48jPWex3f498DnvYF2pLdMcuLT+422oVE9qE5+MAinkIqm6v1+iMXLizqVVa0d4sWUTWc/lgM0UG1qqflrf1nGSqbuZuZXM64D5EjryzbWJLQB72bpd8iXqQxI9kTBQ34uMeJR6cgZDSVFuyKZAlYsAoPqpA75DC4aRPF57IS5eB53tOGiEiYs0QI0OVLXaHZmnlodLZQwOXXYbBldp1OqWK9dtqGzHxuaQutVCfh6ldT2WGjQm80Dh1l/yz3LjxJuWj3GYYCgpgLxGDbzzm3zVrKil7tKfJmgEOHmkAYSzDYXA+59ep/BfWRVAbck7mkTMd+0yLoqasq4CvHqDJwYqqM34mfmuB0q86kKn+g6FNQ/27c3AznmICz86s0TqUT9+x6ovUN62P1wLQGhTcZ6VxQczLwiD/ZkFWxtVEr6ugTtJfaTgV/faTE2K4qbiTOwaDaiwXMuYH+tN9hRZ5PGFH8MKwWHNsUuwf495LzAVe4KYSYzTscysYUDQ1aJPxKKUMdiwIGEoiS/8/Mq4wZWnjQ0A1JIctdph+vV8DxoKkBdav1jl0XAObwZ9nNgSYPIyxzQc7o6p+7OWJvhqrzdueFqy30fpNVJDvxCePg7KnZhZ48FfH8/RwMn4Cf1u7vauVQmj1glLzh9wNnBDcTWMG0PFapp2z5llxa88ayuGJRNPqiamsA0fw19S7CgKjcb9JnScXSOlZf5vl/Cr60Ad5RIpK0zrNDm1VhYh/Rq3hRCtrAM+BhdtmGfvKA5ymFRyHAN89BTzQP9JTcshO7O1+1P4UKsxe353rF7QK9Qeto0yW83hf6slkDTHGfhcVVX3vKG9NHumpbLAbvDfYZWgzG8NOdANc3bQ5KPBOTL0RWhayqDR+fizlrd2X0vzBdSERKoc730yKQW7HGYDktie3mcAOJLSBWfwhLOBCECHsxXwsr5BF3fg3TbrsEq5sixbU1uDLCM6jmtw3vV7vPy06v+PfsuGXksJG/QjdUX1KdKoPkeDx6XqubRsN5bMPo1mXshjpTULRREFS5xmu5kV5XokZ548hS4Nd/PdaINk4ri4Y04XeTVGRu/e8o/U4/f6Vfajs+ArgkHd5WvYyqgxAcTykE9LofzFDu0Ghu3DnK7zIeHxaGdtdntrovKXgWdki5C6Y3wd/Z7Wqv4NLUK8nrzvtZ401TF4nZ9GzvpapD+3C00DEp2R8EyfYZm6LV2T1Rof283JD4Iy9ZTOrO7Tq3A8ckRLN4AKFDKi/e5aJfKOuqutkKat2CNT2K9lTVOp9jGu5fo7xJcuzDX2mjaaMRwVN/+/aG8B2Y2e6uRb6RsZZaB8lLYSGUdEYg2m2aZSY4Ywu6tGoerWgyeBVCMf066rcynzvTcW4NbHvS0LQHfI8KuVknDYl2ooUK6lG1GzimaD87c0llPb9T1mI/HBaEIjbnzvepgLkkacpbTFdI07gX7CpB7tFxzLN07fpNGPWEd1+ZuCvH6Cen9B4UelK0/RqtWQkhYLbHhk+k8pq64iQn4taeCnLOC1mUSGDxxqlm9ajGJpB1DvrjSvwCOck8MUL8cyIVCc3dLOSNiVJDERUnXfs5bu30xTygqAnbhXaVVG4vNbyrpenZFExB8ZUTy1ZhkCo6rCgkRitiFYfl2NTESX23TPvP8n728x470XLYd+C49ZUP0bqBBMOi9d0BDoA167wG9e/OrQuFW66+66jtRIgKRyTh5DLnNWjyLO5o11+Q5aGjDN7Z/VR0eA1cRL9yxrwB+XrhKz5gBPz0oMx65AgiUAmyzLXI8YPpacZjF7RgyMuQ0wgxuS9P6w+E9gh9HvIp55WNAsxk9GUDHcBXb3lBvMiOqyyD6BiOEbddfSWzSUWmXmW014MvAu0g21Z+UQIxmmTy0vR/v2dL7DYsaDRIIMAxyIfqyOAGbDOHYNqvmZOFd771ntk5cLJZntUxua7QyMIhl2S4RtCXVGP8k9w3G+i/Ov4WQnVmTxNsUw3LIOKZQ2ewvcOhe/yJiDUvywG9mor6M8aU29W3Od7k+fZYfrHnPC5DvP3w+QM4r3fkQu9ltQXaWQWeAXXa3Tk7R2x0Rio1MBW77eNWVN2ZQ85D1qN9EYfeYGPRcyPZJ/xpUYSLmwY09kFVTCjyukw/RPfiFZFaPd0u4qX+uPO5tJfxqgAcH0DNlEyuA7WkXsiviZo3qo0RrW2af13z9IlSrq16ovmwpaCFtSviK2/DXW1YF3y4zdLMX4rOQdh+Gu+z30m6iHykwWOstwg94we+DcNQ60slKkvdMSR6JskuSuCGgF6E+fu2dSpF9Y0GLKiokvvIZ+jyQs+UoygNjJDXjUNoewLew+Q1YC64mrr7Q6aVBeHi+rM/jVuZTQ+z/h/tz/foGwr/tnf99L/0fJP/zD4kX87dPPmY1yNrTgcH/ZT9/v4fA/Gf9tJ/17Ue6NFb7Rz888/dm3z8N0r+1W8O//8N//+//VksG40SaUGkKwyROIyWaYWmav39hSIGTeYmlGVXgGY6mBPlHlUJmaElSf5QQFIFSGE38h4YBQxEyLbECeZvjEIUkOYWRFJQgaPJHr4EkKVwS+Ns1hUAZhqEQjKAvEaJRModJBPpnLdk/irIhOPE/IIyCib8WZfs/qfj6ewW4f/X+F0XU/sWhPy3R/1VR9P/30L8b+m/vf9IkYST93zCafgMxjdAQTVH/0CRBxH+jaRrDUQSB/qiH/p0mqaQwCoIQAsISAs4hJKUyKivTpIQK+rUMLMFgBEJoGoLLnCohGKVfQ3ktg6ByLC1p8i+aJIREcgon3wtepDiRoSlKZClV5DRFY3nxvl5TQWD0j3aGphIyK2k4h9ECS3EsQVG4/IsmCf6/Yep9/UWRVGJomUJ/1DHJO0xCpFSK0fm7BhLO8RfGF1meYWhevCv7UxOrILN3pWhBQfAbogso+YsiCYUKmIIRJIWgDE4IgiqSgn6NGCOz/F0vWUA0Tr72jLzOgBLv1JOMxF/OnBbvUOnfCpD9RZFEw/9OkITndIlk75TJpHznmRZQmeUEhRZlRiM5AVPviYfh12Vg7I/YCIbRF9HlaF5CCfrnVP6HIAmnsPJ1p6xEsYQkyxTLXlNAyz9KqgyjcTxDIOg9L3DyLggnaRgrYIhEcJqAEugdivxnYWgBESj0uiGUUCSGFhBKpSSWIlieFBD8Z8Mox8icSF9zw4sUfuf2HvwTJgqkfCPD/0aQ9GG7b2b8F4puPX8VJC0fyn2hHNsr4yVlUBXbvOKFxOS6Kb16bmSzmzhWnndbX9yw7sDbCwoonEt5YYNj+a1kaUvvtebKO57D2V+fM27Vh74WxkaQsvgGmBcp+gDg8aYG09ToAe3zfsm75QDohITwjDCKJxHbY7phA4qJBcZsejliCqLMR0S2ko6fFUtbxGeQnf1VsZBMxVEkYJYAn1H9wFKFBYLKXE35o6Is0W6beOTh0Aqhvys6bUDQv5uYalr01wxrXggPd3Qhg9ZhGK06/iFa5zlPIKOl59HdJdP8TvWnhsHixByPjgoWl0Xv3k6lJyNFFt4ukRs+L7gxX9wwP+Yi5liW5lrgRRRSZXBhFmFRFBQcCIdmqwM5tRm45wtTRbr2jNnRmWLnt0JFF3kzC6euLdWDEvK4TvhKGaoN2XcWiCOIkQ2fCTAuegb70Ov01TCxzoYPZqqxJznL+0siy1gr0DdiBMjPqUnSZMHSBaJJMD/gtxaNB9BOB6pQgwuiy23IL2eE1OJT2hLlFkmqHYPRQkThAJYWOVjWItHBhbAIiP1VncjZR6+rAyJ9PmTlollKcIO/Qk20tbJJUrhUUfkGN/LsN83LxlMu9SmIAzyrwrLQJlVghz61WE34G5i+ZpxjRz8/mImFTAa187vWYjzIlXq4XYzUY8dupK/jNVNKd8oirS1NcKI2QIMeTYscImRptoJlmNLPHUGq6XsLuUZZFQrMjaDE+vTd9ysSCOjKkpF4qYmPFXU2labK/ebjnOFdb+SqHbtcNXkTvQkXv3/kYTipXba52KfM88OEFLoVPP8ixeiD455yW/N17JikwuToYSNvql2/xmnFpNg0tkKay75tlLrQGTBMEfYxKyioKxussxp7iTnWG9OOeM2jOvQdxG4vwoXcT4Q7ewq9TVahduwPIVUEHJCrFeThtK7KJfMON2GWQ5W4ms5RIIQPjsx3taMZ/ruzE5EjopwrA99okyO+WoCMvuhYoxtc1j60IGfknVTWQGz3Ymi6SnGjPqbfljPU1JXOgPAzJ67dNO0wNOda+t5mMIHwh0qgTh4L4kN9XdZnG7rVakVtdB0ceMy/3adajFh3opdguNtgTZNsTVuMqzg99zbh0kr+mUOwzfuD3I2CDOWGgaqWxL8/bDuRXmPTJxnrdI9c0/p9XqNjUl3fDaPqDpzgHTpzilSIGu6N4z9/QpmmivuHuE+sI6yznAmZGCDFhYABej0gyry4LglfCU5fT5HKuldgAr/LrYe/7vHXpSWrpWwdOulmucMlLMrBOp4xkXIwoLIhoYLwG3458hyfJgV+j6qsd9GjukLg4zYlavZRXMIvX4NZ4jG0+aRrRUZEnzq6vLIJ9a92DDrjlM/oH49yDQstUqYx+Wq6tt3PG+dk16cgWM4HMuyNs31oVi5wuWVgx7YKnbMLejnLzlOlMwI9BmaF9lxrMFWXGVNTx+HdXBWpk5tpzh99rJqyCfDfhctWRM1yc9pMtM5u5Pr9dl9x3KL167M5YdsNp0HBoIxdyfpjDlDHbfCuIVYbGldoAYUGKsivuArD7jFTCeY33qvuL79fPof2cbB03qho7e96KVdMPlxRcCFTV1ptM2TUesL064R4soVMiagfCcKc612uhxK9dRV7T/6wiPuZTR2ePqAVTaR5BQ9qkebWU0DDiD49qVrxvVYVxJdfyRYJeezKwLZhgv1i/ezgoALiquFLNppvHrh2A/gtYEln/1TExhk33k1Gl/szCkYIdl+w0Fu/xKMsuhvthzaeSQUU+YQQa8Tj0SUCl1/3Beso9BaDLKQNdTeox76SxWTWzMH01yiQNBYiNbzSS30DTnolw3qIkpfuMhSeYdfuB7qohdqPhV/7N2EMTEfGAmTHqUM+LjDHlJ/5+jXdno9AXJfFN1Kl4pQvan7TKRV18gk+FRK/zvfL7XDwUvdjS0aUePgjQhDlTTXqaItEuYATtVZX2gUTd2MpXo9MA8QPD/BQNE6tW2aLowJ/EzsIMTO/8eVrvA3zdxDom3x0+a6vnzcXvIoACRJf+anEftzfwHculO9QbTU2INlnGLCftCaUS5mzBNzndnYYBMmedY7K/cGQresaNL/Y3GfKp1V3IRA1J0F+siADx9UTUX3MwodqgJUxUm6tiacSDVN1brSHeyfMJMML9qa765bHgzPFzeh63LphBOA11JV4pI2mJgVgVeGWIZorHa/rvkBQx34oF908OjNmq1d4FtesJhw4H/Aswh4rnCMWCREB6an6Xe/XiG0AIOa40RjEDjEgfrN80n9DmOOZP1VpZamjRrs5bP+K+xUP+ZRZB194r2xzf+ct+vP4GI+21fCuswCR9qoLjrKfzfYzgUgGjzBqH4X4NlB2P8rrII7feFOU0kYdfTRIS20f29TfsDXuduxp9MPbYl1U/IYNVPMh7HEDAyeQyzbqh3H6+UHb6QUo8xLua6Qzt3NU/MDfYfknGVC9I1QwTzgLYPCgDPvACjOIFHt4zyY7qUY3kHyjaccieaQQLO0kSVEuRs98rlWajt95SMeCxSADr6Pjk2q8xiojSWkXLr0LJewVpJ9dRZhxX7BqQsBs07VQXEFdLfHVK3rDoIvPG8qwIV3utrG7y7gkgAbqfl6hh/m4lVQfGK87F6oPHJBpDnpuIYjIrb8IASpLRPU7z/bYBkYd+NytyVt9rjfeKhDqQ4F1Qo1F95bovR4xQ/LnVC41QJoNGpbMbi9exiZJCIRJEDsQJ0MLokIEqHNM/S7mKSzb6MdVLuuf1wXG00PUoOXd8iSkVLM+lD57nh4y5fD2KukFMgh9lUYC41vsF56uHZtp9S447idTf2BlQft9VDR92Levj4upZRXboBH7JaydrpHsPq6mjkknFH8PLJSNpEi+7uDcWlgA26kGS0tdZ9yhbcJUFrLWtintWapHuak3ytRgfF5h0jT3dYtUOEsc8eeu15SyEFTr3PYwplMDD06cPdcoqd+Md9jqF5oKmxjI/54XZo6MDjgvGAib73OPQU00DXj3Z2u2xJCPsxCuPHa8IwpP8lXihSxGoRRPhNk7137qqum3n3VTvfVYVx4ZIbd2GxTnW6KuyAOxrpGo7H40j3lFhPmFBXBj38jhvZgAm8VzAHDBNF9+xHY7DgSxNmpmYv62ByLdz9jyNO1xHT1cyfa16RCTyhuBtHt9+rlw4Zmb+OK1QrwLBemAve/mXl2FbaupOW9KplHaIcztgPy822jVVjLCxhGbyuAZzzqvD01b63tjO3St0UrjZjbT9itJfBE6aPwu13EO1QwDDanAL7xcgXw4Jj74uBgS/GKfFpj3MNVVvRg8LCKQfR9uvdo7iJfS4pg4ZR0OXzTpBFLXeRfRFwsaPTP3YjHL/yTSrpgXfJmE7O67JlEIpoHtStWuA6PGRHWhXHIcCZCIzgtOaovw6qJ3TNbIbsiQy1cKlCOHtDG1lyt5MZm90ZwQH9s2zW9M2TTQRNbUMRW94cRXf15qelXdsIkfwtG0W8Y/9N+k+nZKk7p0Mja9yQbDsOdmW0dDOs+2T21+3cEK8qgYr38eQt4RmWfRn7jjWQzS2QYwN2bh6YgUKe3hZ0eE5QhRtFQXv40eD4zpeBHKTchku7sixealD2oDWDHmYyEtX8+3L1zSIssNcprGWvgAvcEnISsAHF7kh+PVEz5XXtEYyJgEoPhqZQAD3EXycsIo1SS8EyvesmFmRwvcnT92ZosdN7XMPuO5Uqt7RWmq9gJsc2HUXilqljRc6uwGNsL6uoNHx93IQuW4D+zvdPqsYxWayFGhKShSWVikw8SVdNXyIgrBNKoFzkFQ1c+FYYMA4iM6kdW83+lCN4qUUjsl4XhDyxbzAvlbosI2OXTDx+MyflH+FYotM57US38mQjV6mEMat1JT5QI7jjlShQeRE1Iku1C06VRHX9VuXPBX9ntpSLVcKGLU3vErf4i96oWEx2FmZlRAVD7mx7CcsLiesfZF6+ec1ia1FiuGlr9taESGyb4Wlxq2bf3AVO/BqsHaNGPUBjzpv4qhxM5l+fS/IIaS/lkMRQYSHKMUc11rg5saVrSQ4Qxc2lOF3VGClzSGYXUoh7Tby3t2hEaIkkTncuUY75KCbiyqMbHttuI9aarY++XIECRh1ihRp/agCwpQIEmT+0GQxP6QyL7lBUF3M7iFBDi4rFkAxDd24PRPsdi8Qw8OxPXXxbmYzmXhJYbNYCOo/cn4SkD0voIzvSIGkJ5oeEw5A2uQVUOXH3pxBcia7iUYw27kZnwKQL/mJQ6W+SKWdeJmL4TQYawKmFilxTAhVZZBIVb5SgerwseYxd2Rm03lfCGweqHf0mHJ4+bNeCHAm0/tsZOw0hBF5U/NcIso93n9TaMBnueexhrRXoSUj71tGPgwxS5nFOgf4XOAI4UwDdcUQbo5Zpr15pOMZl9uzQctVvjLRmnTkKwE6jc1tcqAkoDGF0SzbLmbXT5aHDU3UjopNkIG+2y/6PrIDPX2n5PnRLlq+K7DjU593oE3CKJwse/HaHDHRikmCG+fwVzVKpm1hEd363yKuxPPG9bACni51SFj0Y17ob3gUQQ5oLH+qba7tb8Z7rFp5+So7125zc/6QAShMqBU/xii4Q9dYvofglRUS6Ok+3cCtADMtqBBkpEtDarYIQA/2cvYFqn9Zv3QHPy6mIGzUab7bfLoul4Qv6zPJT3q4Af4UfiQl75X5E10HlEDSE6wDXFiDeqeWh78qfVqoFYLVrIxUHlArfToiJqeHRWwxmOHjBBkqqLjBmVvT4GeD3mdxP4YHX3pb6Pa0HQ9yQDZx6m/RieuiPrv2J9zo1snLz3i+dXo0rAnAf98HSAFqDeWYTjSs229rEyRSU6Fuf2KqT/YjSlGtaGgAUfQOq5jsJaqVPMz6MZtQqHvgrJW2nOgKyQq0PFpyM+nSfOrFX+9yMZa3BIpm7O7kdS2YPrOjcNRZqJjJJh4ZyjmlpWvk2eFd3Gs8O1BL3EbD9f2qW9qwrETpA+e7MDZDwXDPdPH0WxrhSS3kRegddC13yU2NFxNvNm2d5BAtBdP59k9OrVW+5neH7EA45+2FVnuQr97qTzWyyaKKtZdwJop2c9IB7oPobsySl6E+cxEQ2/clne8a0wzUUyFwF5NxK4tE5f78ZAPsSluswFr9HLMBgofSQSACSubXh5LuIW6aMpsCaC09HNJ45nqhVMbeB2JfWmpkWyidzfFaPEGG7Qla9ST7nc8YmiXHr1XqTPvfaNwLdbXUKt8yyWx3Zn/FE/efL4OU/4OTjtGF9ciYMwlYvsSlp/yDZQv1WlD7dtDpQq1csigefsGox2f4TO5UDLAIg/1DmAqk463gdbl6VdHMWKAzL2tSvzNy083bR8Eyq68xnqGcfo2xmr6xq1Iyi/s1JofBy9KDaunWZsBhUAee05urZ/b6nzGRldVxXcP3WSryfooTqdH4AslyKYq1xkYNKO4+RRg9GEZphcVJYIImt89aJFOQwWzj8ZTvtIWkisK1opW/CbGmeBL9MdsIwoYskwbV6Tn2llRk+gCYo6ZRY8UFtdO8sY4n8awlrx0HXMSKkidnLu01aOA1/7Omwg0psdoJixc4++OqnP63Bfk8pTl5neQlxGgmwLhOExaBDf8kKGWkwTRt3D0Wwsia6vMf7/thapZ3zFBA+gwdMtd3NYlPPUhfge91Wm8jmCV/DuKwbd1MwSa7t0uQefOTKHhKUjq28Drm+blBSE2tZSOSGAJtOxvR81+fnGSIxZ41yf30aEfcBFsJXGSMmz0MxJHjIqD+OEaV4MGDtTN25VIAybEF1dtpv7IMbINCoRByDSR8jkHauWT4sKcRIcFZKCWqwkhY03KI6C9XM2MbgRzBtA33/hRG0dVPIhIy1g32+3EhevCtoL+i4WVXGa4ze4tdOUmsDDEjwsFtaAQAFVIm9e2gjdcec42JQdUisyM+GZE5E+Phj9stWF7K8ZLkj+FSBeOpmh+IVDG0mjtUBl15OMpi1Srg1m1/3xRIyo14qO53+whyHY1i2AYhpIhFKJfTdZwZ4pCUdzbani8wrSojjQDtUjeohMgbp8jyttgqgFbaSZu074GI05CV+FuT8tvfzeUtUQzobqE9NGlw7JSLSYu8tgwEYFwG1H27U2D7hKgskQtoVb2ylNd5PZyuMLhU/r8gJlbw5HKBZjdrYioKt7qtyiLHOuBFYOyRyIHrD6az4NVwHXSjBP/2XsCRa4eEAEUjt/P0vwvVIKhM4kh3UZv/rKAJb0h3Iy0a2nH4QFS2K074ODTsCFFshtHeiltLTU+CgLqjcqnd4bNmGzE2JBtGmlCyUcMp2qWbSQpKFOxtd82556VKcj3qj95+eJWsrVk4MlFe2AEJkAwJhy9ZKhnFonh0uIIpu9gvRxcQ63ZMl15cxH4tmhE9HJRilQI320or31M+/pcoEYeQOA9SfI1ERXxqSoCYS4dus4i0zaQk4GJ0dZDeAeKLhLQ4YYd7TPXyhrtLpsz0Q/FQzyKAvxs9tQ1ACyTGipZnHmC6NSZK7N4bdeBAC+KZuXMMrWFL3F5jRMnGPdqinfv6S1ANNLb/VO0+AEKzmnqQAVF/3M56YUmWPrimqymPsdhWS8r1QbzrHv4BbHYL5kQm5weujmcX6OsD6MMCvArkwAqg0ZYclJpRqNbxGvANarVTt1x5n7kyIDZzU2hp+I+WqUu0rt4GZAJlVyNPm/6Cn1USbdnM+54JBiDZsDIzS1qgkysJxCPI2kR8x3sm6GBNyrgmn/2e4yJ/PRLki2encu04o2SwXfYx0rzpbWQ6N6O416gXLFyGIdCnOU8CaO3Q02g819SlMq31xLhS/x+G4V4PEkCG6Z/Wr3oVGsRR+XXzNwaQInFczcTYG6qn372+NJCOE+sxI3EXd0397EqnyaTcjXYQH1aTnHDSc5HBbL4ZVadVdPiSsOXNvdxPcW0bunhVit0+7jannx+PqJF5FPYXu6OoWMmt8NnsdRUK9szsF+H+fGF6TEtatQP+XaKe8tzuVaG08pcYQmBsCBe84DL8/yq8XnRe0QZZkAPGWPQfRMjpghTuIGAVlhuVx+gRkhWE2VS1mgC9JYPzfwcmsWHfHaI1Di2OOqbvLl2FDDxIlNFd5NvyKFUKTSCbvUGUhZOeV0XJiGPhBEn9aEb6VRWV29aBkPN02DaMAYCImWMmkgI7kO+a5uLiZCi9uYG73s4XHTUOzv67JF/VwPLmlqq5Q4xu66VZl9XNa1JugqAsyr8PcljkrPwo9cu+BGXtr3uMQfpx3ttmTFDOGVtOysS+2NcYBOse1L0Zg0laPRa1+LbY5k7XBOA1SqqvHyWc800+QdlTOxNKvB5voByqZk3SCzdopowm1R9kPTu13zSB1myYII8B3bkw+3o3CGBp6hJUTi6pnAQNEcO+WYtC3+nfH0iuGqixJlKj+p4l9fAUuE6eCkCcrFFavNkhlaZ5NgSrhZ/8tX3XmKC6q3M/FqeyyiRCu/FnjltL0/xcoieFWEdxHt5CenzA0INlC0oSbrNtiGOV6NJYMrgWAq+OMYtLs+btEHQ1RR2Fue+mUKtXM6BobjW19j5YjNnjmJeTJurymwaKG0dDZPzeGNv+fx5Lum4AQRzk3lYZbg4EqMOBCF+EAB7LGjH21HwQbP68bPfIsHO3vSwhFunXuk8+0jWoEV0G7KY5H005QRqqN+e9ca2KnJiGRBvc0ARS380z9EigYS1y+9SeO0fdEOdxO4RpOlQ7ZO0xUvtsvBnKjMTNkitOSe8Bhv2wY5kxpoXs4oVkn0dsxPGsD9sVjh/R5x9STSfl4UG7aXjQQ3EpXGiVZy8ttb7WMhdLeKf26c5Kic4ilBLKOFrBPos9xOydW5xHeSV6iz8muUyQWeE2T/EMbj0oWjyCrUJ/oZ7Wd9xfTOAHycG7FAdcJO68PerXQFkAo8Xa54nZkFScngYNYtd4tgTPY2MoMJvyKWUr4AuofYmeSQq/VRycIlvpHxUSkxMFjeiCD/iWVY68wVIcJ2TeV8tjoy1PXuJlJ5QMWLiDw2Hc4/wlH08jgfKcfJZp6ItMrvWMlfOdTWcxQVrS2RymVldyNqEGfVjSdvn/use/3qun49wGv84/h8/38/H+nuxEevRub/dD/D/VT8M7p1XcP7+k34K6E/Rrf+4r/D2w1jR37/+vQ2EP+IBduL/riH492KorIQSsiBKMs8otKSpMsVSKKdIOk1oBC4yoiAwCKaTIivgsoDgMqWJhKQwioYQBEP+qbBWmaZoiSUZlVBwluAUiqd4gtA4gmUQiaUYQhd4UlJkSuZlVtBQmiIJgiVJASHYO3T5f/3P//n//L+K2cJD'
CLASS = ('Unmarked', 'Target only', 'Other only', 'Both')
PALETTE = {1: [67, 204, 204], 2: [249, 118, 163], 3: [225, 187, 64]}


def records():
    return json.loads(zlib.decompress(base64.b64decode(BUNDLE)))


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def norm(a):
    n = np.linalg.norm(a)
    if n < 1e-8:
        raise ValueError('Degenerate mesh direction')
    return a / n


def clean_sample(ct, xyz):
    bounds = np.array(ct.shape) - 1
    valid = ((xyz >= 0) & (xyz <= bounds)).all(axis=-1)
    return np.where(valid, sample_ct(ct, np.clip(xyz, 0, bounds)), 0)


def arrays(record):
    with np.load(io.BytesIO(base64.b64decode(record['meshes'])), allow_pickle=False) as meshes:
        grids = {key: meshes[key] for key in ('VC3D', 'R103')}
    for key, grid in grids.items():
        if digest(grid.astype('<f4')) != record['mesh_sha256'][key]:
            raise ValueError('Registered geometry changed: ' + key)
    return grids


def surface(ct, mesh):
    rows = np.linspace(2, len(mesh) - 1, 5 * (len(mesh) - 3) + 1)[None, :]
    columns = np.linspace(0, 8, 49)[:, None]
    return np.rint(clean_sample(ct, points(mesh, rows, columns))).astype('u1')


def nominated_features(gray):
    # Sample along a regular half-row lattice. Compare the centre against both
    # flanks ten render pixels away. Extend only uninterrupted runs of one sign.
    scales = np.linspace(0, gray.shape[1] - 1, 61)
    values = np.array([np.interp(scales, np.arange(gray.shape[1]), line) for line in gray]).T
    winner = None
    for polarity in (1, -1):
        center = values[:, 10:-10]
        contrast = np.minimum(polarity * (center - values[:, :-20]), polarity * (center - values[:, 20:]))
        history = []
        best = (0, 0.)
        end = None
        previous = {}
        for row, sample in enumerate(contrast):
            current = {}
            for c in np.flatnonzero(sample >= 10):
                options = [(n + 1, strength + float(sample[c]) - .75 * abs(c - last), last)
                           for last, (n, strength, _) in previous.items() if abs(c - last) <= 2]
                current[c] = max(options) if options else (1, float(sample[c]), None)
                if current[c][:2] > best:
                    best, end = current[c][:2], (row, c)
            history.append(current)
            previous = current
        if end is None:
            continue
        row, c = end
        chain = []
        while c is not None:
            chain.append((2 + row * .5, (c + 10) / 6, float(contrast[row, c])))
            c = history[row][c][2]
            row -= 1
        chain.reverse()
        rank = (len(chain), sum(v[2] for v in chain))
        if winner is None or rank > winner[0]:
            winner = (rank, polarity, chain)
    return winner


def slice_images(ct, grid, rows, cols, index, annotation):
    # Flat plane through the marked position, normal to the local trajectory
    # of the texture trace. All basis vectors and samples are in native voxels.
    xyz = points(grid, rows, cols)
    p = xyz[index]
    along = norm(xyz[min(index + 1, len(xyz)-1)] - xyz[max(0, index-1)])
    width = points(np.gradient(grid.astype(float), axis=1), rows[index], cols[index])
    width = norm(width - along * np.dot(along, width))
    up = norm(np.cross(along, width))
    g = np.linspace(-10, 10, 101)
    coordinates = p + g[None, :, None] * width - g[:, None, None] * up
    raw = clean_sample(ct, coordinates)
    plain = png(raw)
    layer = np.repeat(raw[:, :, None], 3, axis=2)
    if annotation is not None:
        zone = classify(coordinates, *annotation).reshape(raw.shape)
        for key, color in PALETTE.items():
            inside = zone == key
            border = np.zeros_like(inside)
            border[1:] |= inside[1:] != inside[:-1]
            border[:, 1:] |= inside[:, 1:] != inside[:, :-1]
            border[:-1] |= inside[:-1] != inside[1:]
            border[:, :-1] |= inside[:, :-1] != inside[:, 1:]
            layer[inside & border] = color
    return {'raw': plain, 'overlay': png(layer), 'point_zyx': p.tolist()}


def one_case(record, ct, labels):
    if digest(ct) != record['crop_sha256']:
        raise ValueError('CT crop differs from the registered input')
    grids = arrays(record)
    rendered = {k: surface(ct, v) for k, v in grids.items()}
    for k, gray in rendered.items():
        if digest(gray) != record['render_hash'][k]:
            raise ValueError('Surface pixels differ from archived CT for ' + k)
    mark = nominated_features(rendered['R103'])
    if mark is None:
        raise ValueError('Archived feature is missing')
    _, polarity, track = mark
    rows, cols, _ = np.asarray(track).T
    if not np.array_equal(rows, record['rows']) or not np.allclose(cols, record['cols'], rtol=0, atol=1e-8):
        raise ValueError('Recomputed track differs from the registered image selection')
    annotation = None
    if labels is not None:
        if int(labels[64,64,64]) != record['target_instance']:
            raise ValueError('Registered target instance differs')
        annotation = masks(labels)
    grid = grids['R103']
    xyz = points(grid, rows, cols)
    n = len(track)
    indices = (0, (n-1)//2, n-1)
    classes = classify(xyz, *annotation) if annotation is not None else None
    return dict(id=record['id'], length=float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
                polarity=polarity, n=n, table=record['area'], target=record['target_instance'],
                texture={k: png(gray) for k,gray in rendered.items()},
                xy=[[float((r-2)/30), float(c/8)] for r,c in zip(rows,cols)],
                indices=indices, classes=None if classes is None else classes.tolist(),
                cuts=[slice_images(ct, grid, rows, cols, j, annotation) for j in indices])


CSS = '''*{box-sizing:border-box}body{margin:0;background:#f7f5ee;color:#192e33;font:17px/1.5 system-ui,sans-serif}main{max-width:970px;margin:auto;padding:32px 26px 64px}h1,h2{font-family:Georgia,serif;line-height:1.1}h1{font-size:clamp(32px,5vw,54px)}h2{font-size:28px}h3{margin:14px 0 5px;font-size:17px}p{max-width:77ch}section{border-top:1px solid #b5c0bd;margin:28px 0;padding:12px 0}.sheets,.cuts{display:grid;gap:12px}.sheets{grid-template-columns:1fr 1fr}.cuts{grid-template-columns:repeat(3,1fr);max-width:774px}figure{margin:0}figcaption{font-weight:600}img{display:block;width:100%}.map,.cut{position:relative}.map svg,.cut svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.cut .marked{display:none}label{display:inline-block;margin:8px 0}input{accent-color:#126263}.overlay:checked~.cuts .plain{display:none}.overlay:checked~.cuts .marked{display:block}.note{color:#4b6264}a{color:#006969}code{overflow-wrap:anywhere}@media(max-width:620px){.sheets{grid-template-columns:1fr}.cuts{gap:5px}main{padding:12px}.cuts figcaption{font-size:14px}}'''


def card(case):
    pts = case['xy']
    inds = case['indices']
    marks = ''.join(f'<circle cx="{pts[j][0]*300:.2f}" cy="{pts[j][1]*96:.2f}" r="2.4" fill="#ffb849"/><text x="{pts[j][0]*300 + (3 if pts[j][0]<.93 else -11):.2f}" y="{max(12,pts[j][1]*96-4):.2f}" fill="white" stroke="#142327" stroke-width="2" paint-order="stroke" font-size="10">{letter}</text>' for j,letter in zip(inds,'ABC'))
    path = ' '.join(f'{x*300:.2f},{y*96:.2f}' for x,y in pts)
    drawing = f'<svg viewBox="0 0 300 96" preserveAspectRatio="none"><polyline points="{path}" fill="none" stroke="#ffb849" stroke-width="1" stroke-dasharray="1 3"/>{marks}</svg>'
    overlay_available = case['classes'] is not None
    blocks=[]
    for j,c in enumerate(case['cuts']):
        # Four small tick marks frame the point, leaving its intensity visible.
        ring='<svg viewBox="0 0 100 100"><path d="M50 42v4 M50 54v4 M42 50h4 M54 50h4" stroke="#ffbc4b" stroke-width="1.0"/></svg>'
        label = ' · ' + CLASS[case['classes'][inds[j]]] if overlay_available else ''
        blocks.append(f'<figure><figcaption>{"ABC"[j]}{label}</figcaption><div class="cut"><img class="plain" alt="CT cross-section, indicated point at centre" src="{c["raw"]}"><img class="marked" alt="Same CT with annotation bounds" src="{c["overlay"]}">{ring}</div></figure>')
    case_id=case['id']; field=case['table']; typ='dark' if case['polarity']==-1 else 'bright'
    result=f'<p>In the full strip, target area changed from {field[0][0]:.1f} to {field[1][0]:.1f} voxel². Other-sheet area changed from {field[0][1]:.1f} to {field[1][1]:.1f} voxel².</p>'
    controls=''; identity=''
    if overlay_available:
        counts=np.bincount(case['classes'], minlength=4)
        identity=f'<p>Of {case["n"]} sampled positions on the R103 trace, {counts[1]} fall near the target alone, {counts[2]} near another sheet alone, {counts[3]} near both and {counts[0]} near neither. Target instance {case["target"]}. The test uses the registered two-voxel annotation tolerance.</p>'
        controls=f'<input class="overlay" id="k{case_id}" type="checkbox"> <label for="k{case_id}">Show annotation boundaries · cyan target, pink other, gold overlap</label>'
    else:
        identity='<p class="note">The archive preview uses CT alone at these marked points. The whole-strip areas below come from the original sheet annotations.</p>'
    textures=case['texture']
    return f'''<section><h2>{case['length']:.1f} voxels along a {typ} trace</h2><p><code>{case_id}</code></p><div class="sheets"><figure><figcaption>VC3D surface</figcaption><img alt="Unmarked grayscale surface before correction" src="{textures['VC3D']}"></figure><figure><figcaption>R103 surface</figcaption><img alt="Unmarked grayscale surface after correction" src="{textures['R103']}"></figure></div><h3>A–C on the same R103 image</h3><div class="map" style="max-width:650px"><img alt="Position of the candidate texture line" src="{textures['R103']}">{drawing}</div><h3>Flat CT sections through the marked points</h3><p>Each panel is 20 × 20 voxels. The small crosshair surrounds the point on the mesh. The section is perpendicular to the local direction of the trace.</p>{controls}<div class="cuts">{''.join(blocks)}</div>{identity}{result}</section>'''


def document(cases, preview=False):
    intro=('One previously published example with its archived CT. The two longest candidates appear on the separate comparison figure.' if preview else 'These two traces ranked first by uninterrupted image contrast across 28 archived R103 surface renders. Selection preceded comparison of the sheet scores. Each surface is shown without annotations before the marked duplicate.')
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>R103 · checking a texture trace</title><style>'+CSS+'</style><main><p>PHerc. Paris. 4 &nbsp; / &nbsp; R103 surface correction</p><h1>Follow the trace into the CT</h1><p>'+intro+' A bright or dark line is a candidate papyrus fiber. The reference marks individual sheets. Anatomical identification of a fiber requires independent tracing.</p>'+''.join(card(c) for c in cases)+'<p class="note">CT from EduceLab-Scrolls, EduceLab / University of Kentucky. Sheet-instance annotations and VC3D from the Vesuvius Challenge. Areas refer to the entire strip. Trace length is the 3D arc length in native voxels.</p></main></html>'


def reconstruct(data_dir):
    cases=[]
    for rec in records()[:2]:
        region = rec['id'].split('_seed_')[0]
        crop = tuple(slice(v,v+128) for v in rec['origin_zyx'])
        raw = read_cube(data_dir, region, 'volume', rec['volume_sha256'])
        labels = read_cube(data_dir, region, 'mask', rec['mask_sha256'])
        cases.append(one_case(rec, image8(raw[crop]), labels[crop]))
    return cases


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    payload = document(reconstruct(args.data_dir))
    with args.out.open('x', encoding='utf-8') as f:
        f.write(payload)
