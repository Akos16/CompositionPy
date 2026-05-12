
#comp = norm*(a*arr1 + b*arr2+ c*arr3 + d*arr4)
#0 <= a, b, c, d <= 1  ugy hogy a+b+c+d=1
#composition[0]["18"][260] [0. txt]["lgE"][index, txt 1. sora a 0. elem]
class compositionHandler():
    def weightedComposition(self, composition):
        comp0 = []
        comp1 = []
        comp2 = []
        comp3 = []
        tempComp0 = []
        tempParamComp = composition[0]["18"][180]
        '''
        for i in range(3):
            comp.append(composition[0]["18"][i])
        for j in range(len(comp)):
            result.append(comp[j] * 2)
        '''
        nums = [0.75, 0, 0, 0]
        '''
        for i in range(len(nums)):
            for j in range(len(composition[0]["18"])):
                comp.append(composition[i]["18"][j] * nums[i])
        '''
        for i in range(len(composition[0]["18"])):
            comp0.append(composition[0]["18"][i] * 0.75)
        tempComp0 = composition[0]["18"] * 0.75
        for j in range(len(composition[1]["18"])):
            comp1.append(composition[1]["18"][j])
        for k in range(len(composition[2]["18"])):
            comp2.append(composition[2]["18"][k])
        for z in range(len(composition[3]["18"])):
            comp3.append(composition[3]["18"][z])
        comLSum = len(comp0) + len(comp1) + len(comp2) + len(comp3)
        comZSum = sum(comp0 + comp1 + comp2 + comp3) / comLSum 
        return comZSum, comp0[180], tempComp0[180], tempParamComp