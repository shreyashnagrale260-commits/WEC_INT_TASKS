public class Stocksproblem {
    public static int FindProfit(int prices[]){
        int maxProfit = 0;                                  //tracks maximum profit
        int minBuyPrice = Integer.MAX_VALUE;                //tracks minimum buying price
        for(int i =0;i<prices.length;i++){
            if(minBuyPrice < prices[i]){
                int profit = prices[i] - minBuyPrice;              //calculating today's profit
                maxProfit = Integer.max(profit,maxProfit);         //taking max of today's profit and previous maximum profit
            }else{
                minBuyPrice=prices[i];
            }

        }
        return maxProfit;                      //Time complexity - O(n)
    }
    public static void main(String args[]){
        int prices[]={23,45,76,765,987};
        System.out.println(FindProfit(prices));
    }
    
}
