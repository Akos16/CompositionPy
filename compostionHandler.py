#comp = norm*(a*arr1 + b*arr2+ c*arr3 + d*arr4)
#0 <= a, b, c, d <= 1  ugy hogy a+b+c+d=1
#composition[0]["18"][260] [0. txt]["lgE"][index, txt 1. sora a 0. elem]
class compositionHandler():
    def weightedComposition(self, composition):
        comp = (composition[0]["18"]/composition[0]["18"].sum() * 0.70 + composition[1]["18"]/composition[1]["18"].sum() * 0.10 + composition[2]["18"]/composition[2]["18"].sum() * 0.10 + composition[3]["18"]/composition[3]["18"].sum() * 0.10)

        return comp, composition[0]["18"]
    