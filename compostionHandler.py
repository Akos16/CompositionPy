
#comp = norm*(a*arr1 + b*arr2+ c*arr3 + d*arr4)
#0 <= a, b, c, d <= 1  ugy hogy a+b+c+d=1
#composition[0]["18"][260] [0. txt]["lgE"][index, txt 1. sora a 0. elem]
class compositionHandler():
    def weightedComposition(self, composition):
        comp = []
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
            comp.append(composition[0]["18"][i] * 0.75)
        comZSum = sum(comp)
        return comZSum