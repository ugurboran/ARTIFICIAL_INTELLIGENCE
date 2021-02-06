//Uğur_Boran_Week4_Monty_Hall_Probability_of_Loses

import java.util.Random;
import java.util.Scanner;

public class montyprob {

	public static void main(String[] args) {

		Scanner scan = new Scanner(System.in);
		Random generator = new Random();

		int user_choice,open_door,prize_door,other_door;
		int congrulations = 0;
		double loses = 0;


		for(int i=0;i<10000;i++) {

			prize_door = generator.nextInt(3)+1;
			other_door = prize_door;

			//System.out.println("MONTY HALL GAME");

			do{

				user_choice = generator.nextInt(3)+1;
			}while(user_choice > 3 || user_choice < 0);
			do{
				open_door = generator.nextInt(3)+1;
			}while(open_door == prize_door || open_door == user_choice);


			//System.out.println("\nBehind door number " + open_door+ " are goats!");

			//System.out.println("You selected door number " + user_choice);



			//System.out.println("The prize is behind door number: " + prize_door);

			if (user_choice == prize_door) {
				congrulations++;
			} else {
				loses++;
			}
		}
		
		System.out.print("The result is : "+ loses/10000);
	}
}





